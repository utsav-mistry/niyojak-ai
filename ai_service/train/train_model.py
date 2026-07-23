"""
train_model.py — niyojak XGBoost node scoring model trainer

Generates synthetic node telemetry data covering three realistic scenarios
(healthy, moderate load, stressed/saturated) and trains an XGBoost regression
model to predict a node placement score (0-100).

The trained model is saved to ../model/niyojak_model.pkl and loaded at
runtime by ai_service/app/model.py.

Usage:
    python train_model.py

Output:
    ../model/niyojak_model.pkl
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("niyojak.train")

# Feature columns must exactly match what feature_store.py produces
# and what model.py expects at inference time.
FEATURE_COLUMNS = [
    "cpu_mean",
    "cpu_max",
    "cpu_std",
    "cpu_spike_rate",
    "mem_mean",
    "mem_max",
    "mem_std",
    "load_mean",
    "load_max",
    "net_rx_mean",
    "net_tx_mean",
]

# Output directory for the trained model
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "niyojak_model.pkl")


def generate_dataset(n_samples: int = 8000) -> pd.DataFrame:
    """
    Generate synthetic node telemetry data with labelled placement scores.

    Three scenario classes:
      - Healthy / idle           -> score 80-100  (ideal placement target)
      - Moderate load            -> score 40-79   (acceptable)
      - Stressed / saturating    -> score 0-39    (avoid)
    """
    rng = np.random.default_rng(seed=42)
    rows = []

    # --- Scenario 1: Healthy idle node ---
    n = n_samples // 3
    for _ in range(n):
        cpu = rng.uniform(0.03, 0.25)
        mem = rng.uniform(0.10, 0.38)
        load = rng.uniform(0.05, 0.40)
        rows.append({
            "cpu_mean":       cpu,
            "cpu_max":        cpu + rng.uniform(0.01, 0.05),
            "cpu_std":        rng.uniform(0.005, 0.03),
            "cpu_spike_rate": rng.uniform(0.0, 0.05),
            "mem_mean":       mem,
            "mem_max":        mem + rng.uniform(0.01, 0.05),
            "mem_std":        rng.uniform(0.005, 0.02),
            "load_mean":      load,
            "load_max":       load + rng.uniform(0.1, 0.5),
            "net_rx_mean":    rng.uniform(1e4, 2e6),
            "net_tx_mean":    rng.uniform(1e4, 2e6),
            "score":          rng.uniform(80, 100),
        })

    # --- Scenario 2: Moderate load ---
    n = n_samples // 3
    for _ in range(n):
        cpu = rng.uniform(0.35, 0.65)
        mem = rng.uniform(0.40, 0.70)
        load = rng.uniform(0.8, 2.5)
        rows.append({
            "cpu_mean":       cpu,
            "cpu_max":        cpu + rng.uniform(0.05, 0.15),
            "cpu_std":        rng.uniform(0.02, 0.08),
            "cpu_spike_rate": rng.uniform(0.0, 0.20),
            "mem_mean":       mem,
            "mem_max":        mem + rng.uniform(0.05, 0.12),
            "mem_std":        rng.uniform(0.01, 0.06),
            "load_mean":      load,
            "load_max":       load + rng.uniform(0.5, 1.5),
            "net_rx_mean":    rng.uniform(1e6, 20e6),
            "net_tx_mean":    rng.uniform(1e6, 20e6),
            "score":          rng.uniform(40, 79),
        })

    # --- Scenario 3: Stressed / saturating ---
    n = n_samples - 2 * (n_samples // 3)
    for _ in range(n):
        cpu = rng.uniform(0.75, 0.99)
        mem = rng.uniform(0.72, 0.98)
        load = rng.uniform(3.0, 8.0)
        rows.append({
            "cpu_mean":       cpu,
            "cpu_max":        min(1.0, cpu + rng.uniform(0.01, 0.10)),
            "cpu_std":        rng.uniform(0.04, 0.15),
            "cpu_spike_rate": rng.uniform(0.30, 1.0),
            "mem_mean":       mem,
            "mem_max":        min(1.0, mem + rng.uniform(0.01, 0.08)),
            "mem_std":        rng.uniform(0.02, 0.10),
            "load_mean":      load,
            "load_max":       load + rng.uniform(1.0, 4.0),
            "net_rx_mean":    rng.uniform(20e6, 200e6),
            "net_tx_mean":    rng.uniform(20e6, 200e6),
            "score":          rng.uniform(0, 39),
        })

    df = pd.DataFrame(rows)
    # Shuffle so scenarios are interleaved during training
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def train(df: pd.DataFrame) -> XGBRegressor:
    X = df[FEATURE_COLUMNS]
    y = df["score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        eval_metric="mae",
        early_stopping_rounds=20,
        random_state=42,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    logger.info("Training complete — test MAE: %.2f / 100", mae)

    return model


def save(model: XGBRegressor) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", MODEL_PATH)


if __name__ == "__main__":
    logger.info("Generating synthetic training data (%d samples)...", 8000)
    df = generate_dataset(n_samples=8000)

    logger.info(
        "Score distribution: min=%.1f mean=%.1f max=%.1f",
        df["score"].min(), df["score"].mean(), df["score"].max(),
    )

    logger.info("Training XGBoost model...")
    model = train(df)

    save(model)
    logger.info("Done. Run the AI service to use this model.")
