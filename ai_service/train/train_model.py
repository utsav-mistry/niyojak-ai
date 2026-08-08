"""
train_model.py — Niyojak Student Model Trainer
===============================================

OFFLINE DEVELOPER TOOL — never called by the production scheduler.
The runtime AI service uses a frozen, pre-trained model for inference.

Teacher-Student Architecture
-----------------------------
Input  : teacher_dataset.csv / teacher_dataset.parquet  (from generate_dataset.py)
Teacher: the Go scheduler's heuristic scoring formula
Student: XGBoost regression model approximating the teacher

Train/Test Split
-----------------
Split is performed BY EVENT (event_id), not by row.  This prevents
data leakage: all (pod, node) pairs from the same scheduling event
stay on the same side of the split.

Feature Policy
---------------
IDs (event_id, cluster_id, pod_id, node_id, best_node_id) are METADATA
and are never passed to the model as input features.

Labels (teacher_score, selected_by_scheduler, teacher_rank,
score_gap_from_best, winner_score) are also excluded from features.

Penalty components (cpu_penalty, memory_penalty, density_penalty) are
kept as features because they encode the teacher's reasoning and improve
model explainability.

Usage
-----
    # Step 1: generate the dataset (run once, or when teacher changes)
    python train/generate_dataset.py --target-events 20000

    # Step 2: train the student model
    python train/train_model.py

    # Specify a different dataset path
    python train/train_model.py --dataset-path data/teacher_dataset.parquet

Output
------
    model/niyojak_model.json
    model/metadata.json
"""

import argparse
import json
import logging
import os
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("niyojak.train")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE      = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR  = os.path.join(_HERE, "..", "data")
_MODEL_DIR = os.path.join(_HERE, "..", "model")
MODEL_PATH = os.path.join(_MODEL_DIR, "niyojak_model.json")
METADATA_PATH = os.path.join(_MODEL_DIR, "metadata.json")

# Default dataset search order: parquet first (smaller), then CSV
DEFAULT_PARQUET = os.path.join(_DATA_DIR, "teacher_dataset.parquet")
DEFAULT_CSV     = os.path.join(_DATA_DIR, "teacher_dataset.csv")


# ---------------------------------------------------------------------------
# Feature columns
#
# Policy:
#   - IDs (event_id, cluster_id, pod_id, node_id, best_node_id)  → NEVER features
#   - Labels (teacher_score, teacher_rank, selected_by_scheduler,
#             score_gap_from_best, winner_score)                  → NEVER features
#   - Feasibility metadata (feasible, reject_reason)              → NEVER features
#   - All numeric signals below                                   → features
#
# Penalty components (cpu_penalty, memory_penalty, density_penalty) are
# included because they directly encode teacher reasoning — they give the
# model a head start on the teacher's decision logic.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    # ── Node capacity context ─────────────────────────────────────────
    "cluster_size",
    # Keep allocatable capacities (absolute capacities are redundant)
    "node_allocatable_cpu",
    "node_allocatable_memory",

    # ── Current node state ────────────────────────────────────────────
    "current_cpu_usage",
    "current_memory_usage",
    "current_cpu_percent",
    "current_memory_percent",
    "current_pod_count",

    # ── Incoming pod ──────────────────────────────────────────────────
    "requested_cpu",
    "requested_memory",

    # ── Projected state ───────────────────────────────────────────────
    # Keep projected percents as compact representation of projected state
    "projected_cpu_percent",
    "projected_memory_percent",

    # ── Derived features ──────────────────────────────────────────────
    # Derived compact features (avoid duplication)
    "cpu_headroom",
    "memory_headroom",
    "resource_balance",
    "cpu_request_ratio",
    "memory_request_ratio",
    "packing_density",
]

TARGET_COLUMN = "teacher_score"
GROUP_COLUMN  = "event_id"      # used for event-level train/test split


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(path: str = "") -> tuple[pd.DataFrame, str]:
    """
    Load the teacher dataset.  Tries the supplied path first, then falls
    back to parquet, then CSV in the default data directory.
    """
    candidates = [path] if path else []
    candidates += [DEFAULT_PARQUET, DEFAULT_CSV]

    for p in candidates:
        if not p:
            continue
        if not os.path.exists(p):
            continue
        ext = os.path.splitext(p)[1].lower()
        logger.info("Loading dataset from %s", p)
        if ext == ".parquet":
            try:
                df = pd.read_parquet(p, engine="fastparquet")
                logger.info("Loaded %d rows from Parquet", len(df))
                return df, p
            except Exception as exc:
                logger.warning("Parquet load failed (%s) — trying CSV", exc)
        else:
            df = pd.read_csv(p)
            logger.info("Loaded %d rows from CSV", len(df))
            return df, p

    raise FileNotFoundError(
        "No dataset found.  Run:\n"
        "    python train/generate_dataset.py\n"
        "to generate teacher_dataset.csv / .parquet first."
    )


def _seed_everything(seed: int) -> None:
    """Seed all local RNGs used by the trainer for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def _gpu_is_available() -> bool:
    """Best-effort GPU detection for XGBoost training."""
    try:
        import shutil
        import subprocess

        if shutil.which("nvidia-smi") is None:
            return False
        probe = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=False,
        )
        return probe.returncode == 0 and bool(probe.stdout.strip())
    except Exception:
        return False


def _build_training_params(seed: int, use_gpu: bool) -> dict:
    """Return the production training hyperparameters and device settings."""
    params = {
        "n_estimators": 5000,
        "max_depth": 8,
        "learning_rate": 0.02,
        "subsample": 0.80,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.2,
        "max_bin": 256,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "random_state": seed,
        "seed": seed,
        "n_jobs": -1,
        "verbosity": 200,
        "early_stopping_rounds": 100,
        "device": "cuda:0",
        "tree_method": "hist",
    }
    return params

# ---------------------------------------------------------------------------
# Train/test split — by event, not by row
# ---------------------------------------------------------------------------

def split_by_event(
    df: pd.DataFrame,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple:
    """
    Split rows by unique event_id so that all (pod, node) pairs from the
    same scheduling event stay on the same side of the split.
    """
    events = df[GROUP_COLUMN].unique()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df[GROUP_COLUMN].values

    train_idx, test_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx]
    test_df  = df.iloc[test_idx]
    
    # MEMORY FIX: We do not need a copy of the massive 19M+ row training 
    # dataframe for evaluation. We only need the test set. 
    train_df_full = None
    test_df_full = test_df.copy()

    # MEMORY FIX: Use .loc to simultaneously filter feasible rows AND 
    # select only the needed columns before copying. This prevents pandas 
    # from creating massive intermediate dataframes in memory.
    X_train = train_df.loc[train_df["feasible"], FEATURE_COLUMNS].copy()
    y_train = train_df.loc[train_df["feasible"], TARGET_COLUMN].copy()
    
    X_test = test_df.loc[test_df["feasible"], FEATURE_COLUMNS].copy()
    y_test = test_df.loc[test_df["feasible"], TARGET_COLUMN].copy()

    train_events = train_df[GROUP_COLUMN].nunique()
    test_events = test_df[GROUP_COLUMN].nunique()

    logger.info(
        "Train: %d rows across %d events | Test: %d rows across %d events",
        len(X_train), train_events, len(X_test), test_events,
    )
    
    # Returning None for train_df_full keeps the function signature intact
    return X_train, X_test, y_train, y_test, train_events, test_events, train_df_full, test_df_full

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train(X_train: pd.DataFrame, X_test: pd.DataFrame,
          y_train: pd.Series, y_test: pd.Series,
          train_df_full: pd.DataFrame, test_df_full: pd.DataFrame,
          seed: int) -> tuple[XGBRegressor, dict, dict]:
    """Train an XGBoost regression model to approximate the teacher scorer.

    The saved artifact is the regression model only, to remain runtime-compatible.
    Evaluation additionally records ranking metrics and artifact metadata.
    """
    use_gpu = _gpu_is_available()
    training_params = _build_training_params(seed=seed, use_gpu=use_gpu)
    model = XGBRegressor(**training_params)

    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "eval_set": [(X_test, y_test)],
        "verbose": False,
    }
    model.fit(**fit_kwargs)

    preds = model.predict(X_test)
    preds = np.clip(preds, 0.0, 100.0)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    medae = median_absolute_error(y_test, preds)
    mask_nonzero = y_test.values != 0
    mape = mean_absolute_percentage_error(y_test.values[mask_nonzero], preds[mask_nonzero]) if mask_nonzero.any() else 0.0
    r2 = r2_score(y_test, preds)
    explained_var = explained_variance_score(y_test, preds)
    best_iteration = int(getattr(model, "best_iteration", model.n_estimators - 1) or 0)
    logger.info(
        "Training complete — test MAE: %.2f / 100 | RMSE: %.2f | MedAE: %.2f | MAPE: %.3f | R2: %.3f | EVS: %.3f",
        mae, rmse, medae, mape, r2, explained_var,
    )

    _log_score_band_mae(y_test.values, preds)
    _log_feature_importance(model)

    # Ranking-focused evaluation: winner/top-k accuracy and rank correlations
    try:
        # Optional ranking correlation imports — continue if scipy missing
        try:
            from scipy.stats import spearmanr, kendalltau
        except Exception:
            spearmanr = None
            kendalltau = None
        # Attach predictions back to test_df_full by index alignment
        test_df_full = test_df_full.copy()
        
        # We already predicted on X_test (which is the feasible rows of test_df_full)
        # Avoid re-predicting by directly assigning the predictions back
        feas_df = test_df_full[test_df_full["feasible"]].copy()
        feas_df = feas_df.assign(pred_score=preds)

        # Winner and top-3 accuracy
        winner_acc_n = 0
        top3_acc_n = 0
        events = 0
        spearman_vals = []
        kendall_vals = []
        for event_id, g in feas_df.groupby(GROUP_COLUMN):
            # true winner node_id where selected_by_scheduler==1
            winners = g[g["selected_by_scheduler"] == 1]
            if winners.empty:
                continue
            events += 1
            true_winner = winners.iloc[0]["node_id"]
            preds_sorted = g.sort_values("pred_score", ascending=False)
            pred_top1 = preds_sorted.iloc[0]["node_id"]
            topk = preds_sorted.head(3)["node_id"].tolist()
            if pred_top1 == true_winner:
                winner_acc_n += 1
            if true_winner in topk:
                top3_acc_n += 1

            # rank correlations
            try:
                if spearmanr is not None:
                    sr = spearmanr(g["teacher_score"].values, g["pred_score"].values)
                    if getattr(sr, "correlation", None) is not None:
                        spearman_vals.append(sr.correlation)
            except Exception:
                pass
            try:
                if kendalltau is not None:
                    kt = kendalltau(g["teacher_score"].values, g["pred_score"].values)
                    if getattr(kt, "correlation", None) is not None:
                        kendall_vals.append(kt.correlation)
            except Exception:
                pass

        winner_acc = winner_acc_n / events if events > 0 else 0.0
        top3_acc = top3_acc_n / events if events > 0 else 0.0
        mean_spearman = float(sum(spearman_vals) / len(spearman_vals)) if spearman_vals else 0.0
        mean_kendall = float(sum(kendall_vals) / len(kendall_vals)) if kendall_vals else 0.0

        logger.info("Winner accuracy: %.3f | Top-3 accuracy: %.3f", winner_acc, top3_acc)
        logger.info("Mean Spearman: %.3f | Mean Kendall: %.3f", mean_spearman, mean_kendall)
    except Exception as exc:
        logger.warning("Ranking evaluation skipped due to error: %s", exc)

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "median_absolute_error": float(medae),
        "mape": float(mape),
        "r2": float(r2),
        "explained_variance": float(explained_var),
        "winner_accuracy": float(winner_acc if 'winner_acc' in locals() else 0.0),
        "top3_accuracy": float(top3_acc if 'top3_acc' in locals() else 0.0),
        "best_iteration": int(best_iteration),
    }

    return model, metrics, training_params


def _log_score_band_mae(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Log MAE per score band for diagnostic clarity."""
    bands = [
        ("0-20  (saturated)",   0,  20),
        ("20-40 (stressed)",   20,  40),
        ("40-60 (moderate)",   40,  60),
        ("60-80 (healthy)",    60,  80),
        ("80-100 (idle)",      80, 100),
    ]
    logger.info("MAE by teacher score band:")
    for label, lo, hi in bands:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() > 0:
            band_mae = mean_absolute_error(y_true[mask], y_pred[mask])
            logger.info("  [%s] n=%d  MAE=%.2f", label, mask.sum(), band_mae)


def _log_feature_importance(model: XGBRegressor) -> None:
    """Log the top-10 features for gain, weight, and cover importance."""
    booster = model.get_booster()
    for importance_type in ("gain", "weight", "cover"):
        importance = booster.get_score(importance_type=importance_type)
        sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
        logger.info("Top-10 features by %s importance:", importance_type)
        for i, (fname, score) in enumerate(sorted_imp[:10], 1):
            logger.info("  %2d. %-35s %s=%.1f", i, fname, importance_type, score)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save(model: XGBRegressor, metadata: dict) -> None:
    os.makedirs(_MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    logger.info("Model saved to %s", MODEL_PATH)
    logger.info("Metadata saved to %s", METADATA_PATH)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Niyojak student model from the teacher dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path", default="",
        help="Path to teacher_dataset.csv or .parquet.  "
             "Defaults to data/teacher_dataset.parquet (then .csv).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.15,
        help="Fraction of events to hold out for evaluation.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/test split.",
    )
    args = parser.parse_args()

    _seed_everything(args.seed)

    logger.info("Step 1/5: Loading dataset...")
    df, dataset_path = load_dataset(args.dataset_path)

    # Sanity-check that the required columns are present
    logger.info("Step 2/5: Validating schema and computing statistics...")
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN, GROUP_COLUMN]
               if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}\n"
            "Regenerate with: python train/generate_dataset.py"
        )

    logger.info(
        "Dataset loaded: %d rows | %d events | %d feature columns",
        len(df), df[GROUP_COLUMN].nunique(), len(FEATURE_COLUMNS),
    )
    
    logger.info("Computing score distribution (this may take a moment on massive datasets)...")
    feas_scores = df.loc[df["feasible"], TARGET_COLUMN]
    logger.info(
        "Teacher score distribution: min=%.1f  mean=%.1f  max=%.1f  std=%.1f",
        feas_scores.min(),
        feas_scores.mean(),
        feas_scores.max(),
        feas_scores.std(),
    )

    logger.info("Step 3/5: Performing event-level train/test split...")
    X_train, X_test, y_train, y_test, tr_ev, te_ev, train_df_full, test_df_full = split_by_event(
        df, test_size=args.test_size, seed=args.seed
    )

    logger.info("Step 4/5: Training XGBoost student model...")
    logger.info(
        "Training on %d feasible rows (%d events)...",
        len(X_train), tr_ev,
    )
    model, metrics, training_params = train(
        X_train, X_test, y_train, y_test, train_df_full, test_df_full, seed=args.seed
    )
    
    logger.info("Step 5/5: Preparing evaluation metrics and saving artifacts...")

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "training_parameters": training_params,
        "dataset_path": os.path.abspath(dataset_path),
        "dataset_size": {
            "rows": int(len(df)),
            "events": int(df[GROUP_COLUMN].nunique()),
            "feasible_rows": int(df["feasible"].sum()),
            "infeasible_rows": int((~df["feasible"]).sum()),
        },
        "xgboost_version": xgb.__version__,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "median_absolute_error": metrics["median_absolute_error"],
        "mape": metrics["mape"],
        "explained_variance": metrics["explained_variance"],
        "winner_accuracy": metrics["winner_accuracy"],
        "top3_accuracy": metrics["top3_accuracy"],
        "best_iteration": metrics["best_iteration"],
    }

    save(model, metadata)
    logger.info("Done. Run the AI service to use this model.")


if __name__ == "__main__":
    main()
