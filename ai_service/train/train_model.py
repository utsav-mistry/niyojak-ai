"""
train_model.py — niyojak XGBoost node scoring model trainer

Generates synthetic node telemetry data covering ten realistic scenarios
and trains an XGBoost regression model to predict a node placement score (0-100).

Scenario catalogue
------------------
  1.  Healthy idle             score 88-100  (cold / newly-provisioned node)
  2.  Healthy active           score 78-92   (steady moderate workload, lots of headroom)
  3.  Near-capacity            score 42-62   (approaching limits but not saturated)
  4.  Moderate load            score 40-70   (typical busy node)
  5.  Memory pressure          score 18-42   (high mem, low CPU — potential OOM risk)
  6.  CPU burst / flapping     score 22-50   (intermittent CPU spikes, high std)
  7.  Network-bound            score 30-58   (high net I/O, moderate compute)
  8.  Mixed stress             score 8-35    (high mem + moderate-high CPU together)
  9.  Stressed / saturating    score 4-36    (high CPU, high mem, high load)
  10. Fully saturated          score 0-12    (everything maxed — worst-case edge)

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

# Total samples — distributed across 10 scenarios
N_SAMPLES_DEFAULT = 30_000


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def generate_dataset(n_samples: int = N_SAMPLES_DEFAULT) -> pd.DataFrame:
    """
    Generate synthetic node telemetry data with labelled placement scores.

    Ten scenario classes span the full range of realistic node states, including
    edge cases (fully-saturated, near-zero warmup, memory-only pressure, bursting
    CPU, network-bound nodes, and mixed stress). Each scenario is allocated an
    equal slice of n_samples; the remainder goes to the last scenario.

    Score interpretation:
        88-100  -> ideal placement target (very healthy / idle)
        78-92   -> healthy with active workload
        42-62   -> near capacity but acceptable
        40-70   -> moderate load
        18-42   -> memory-pressure risk
        22-50   -> bursty / flapping CPU
        30-58   -> network-bound
         8-35   -> mixed stress
         4-36   -> heavily stressed / saturating
         0-12   -> fully saturated -- avoid at all costs
    """
    rng = np.random.default_rng(seed=42)
    rows: list[dict] = []

    n = n_samples // 10

    # ------------------------------------------------------------------
    # Scenario 1 -- Healthy idle / warm-up (newly provisioned node)
    # Very low CPU and memory; network and load near-zero.
    # ------------------------------------------------------------------
    for _ in range(n):
        cpu = rng.uniform(0.01, 0.10)
        mem = rng.uniform(0.05, 0.20)
        load = rng.uniform(0.01, 0.15)
        rows.append({
            "cpu_mean":       cpu,
            "cpu_max":        _clamp(cpu + rng.uniform(0.005, 0.03)),
            "cpu_std":        rng.uniform(0.001, 0.010),
            "cpu_spike_rate": rng.uniform(0.0, 0.01),
            "mem_mean":       mem,
            "mem_max":        _clamp(mem + rng.uniform(0.005, 0.02)),
            "mem_std":        rng.uniform(0.001, 0.008),
            "load_mean":      load,
            "load_max":       load + rng.uniform(0.01, 0.10),
            "net_rx_mean":    rng.uniform(0, 5e4),
            "net_tx_mean":    rng.uniform(0, 5e4),
            "score":          rng.uniform(88, 100),
        })

    # ------------------------------------------------------------------
    # Scenario 2 -- Healthy active (steady moderate workload, headroom)
    # ------------------------------------------------------------------
    for _ in range(n):
        cpu = rng.uniform(0.05, 0.28)
        mem = rng.uniform(0.10, 0.38)
        load = rng.uniform(0.10, 0.55)
        rows.append({
            "cpu_mean":       cpu,
            "cpu_max":        _clamp(cpu + rng.uniform(0.02, 0.07)),
            "cpu_std":        rng.uniform(0.005, 0.025),
            "cpu_spike_rate": rng.uniform(0.0, 0.04),
            "mem_mean":       mem,
            "mem_max":        _clamp(mem + rng.uniform(0.01, 0.06)),
            "mem_std":        rng.uniform(0.003, 0.018),
            "load_mean":      load,
            "load_max":       load + rng.uniform(0.10, 0.50),
            "net_rx_mean":    rng.uniform(1e4, 3e6),
            "net_tx_mean":    rng.uniform(1e4, 3e6),
            "score":          rng.uniform(78, 92),
        })

    # ------------------------------------------------------------------
    # Scenario 3 -- Near-capacity (approaching limits, headroom shrinking)
    # ------------------------------------------------------------------
    for _ in range(n):
        cpu = rng.uniform(0.55, 0.72)
        mem = rng.uniform(0.58, 0.75)
        load = rng.uniform(1.5, 3.0)
        rows.append({
            "cpu_mean":       cpu,
            "cpu_max":        _clamp(cpu + rng.uniform(0.05, 0.18)),
            "cpu_std":        rng.uniform(0.03, 0.10),
            "cpu_spike_rate": rng.uniform(0.05, 0.25),
            "mem_mean":       mem,
            "mem_max":        _clamp(mem + rng.uniform(0.04, 0.12)),
            "mem_std":        rng.uniform(0.01, 0.06),
            "load_mean":      load,
            "load_max":       load + rng.uniform(0.5, 2.0),
            "net_rx_mean":    rng.uniform(5e6, 30e6),
            "net_tx_mean":    rng.uniform(5e6, 30e6),
            "score":          rng.uniform(42, 62),
        })

    # ------------------------------------------------------------------
    # Scenario 4 -- Moderate load (typical busy production node)
    # ------------------------------------------------------------------
    for _ in range(n):
        cpu = rng.uniform(0.30, 0.62)
        mem = rng.uniform(0.35, 0.68)
        load = rng.uniform(0.80, 2.50)
        rows.append({
            "cpu_mean":       cpu,
            "cpu_max":        _clamp(cpu + rng.uniform(0.05, 0.15)),
            "cpu_std":        rng.uniform(0.02, 0.09),
            "cpu_spike_rate": rng.uniform(0.0, 0.20),
            "mem_mean":       mem,
            "mem_max":        _clamp(mem + rng.uniform(0.04, 0.12)),
            "mem_std":        rng.uniform(0.01, 0.06),
            "load_mean":      load,
            "load_max":       load + rng.uniform(0.4, 1.5),
            "net_rx_mean":    rng.uniform(1e6, 20e6),
            "net_tx_mean":    rng.uniform(1e6, 20e6),
            "score":          rng.uniform(40, 70),
        })

    # ------------------------------------------------------------------
    # Scenario 5 -- Memory pressure (high mem, low CPU -- OOM risk)
    # Edge case: memory is near-critical but CPU is still fine.
    # ------------------------------------------------------------------
    for _ in range(n):
        cpu = rng.uniform(0.05, 0.35)
        mem = rng.uniform(0.78, 0.97)
        load = rng.uniform(0.20, 1.50)
        rows.append({
            "cpu_mean":       cpu,
            "cpu_max":        _clamp(cpu + rng.uniform(0.02, 0.10)),
            "cpu_std":        rng.uniform(0.01, 0.06),
            "cpu_spike_rate": rng.uniform(0.0, 0.10),
            "mem_mean":       mem,
            "mem_max":        _clamp(mem + rng.uniform(0.005, 0.04)),
            "mem_std":        rng.uniform(0.005, 0.025),
            "load_mean":      load,
            "load_max":       load + rng.uniform(0.20, 1.0),
            "net_rx_mean":    rng.uniform(5e4, 8e6),
            "net_tx_mean":    rng.uniform(5e4, 8e6),
            "score":          rng.uniform(18, 42),
        })

    # ------------------------------------------------------------------
    # Scenario 6 -- CPU burst / flapping (intermittent spikes, high std)
    # Edge case: average looks okay but variance is very high.
    # ------------------------------------------------------------------
    for _ in range(n):
        cpu_base = rng.uniform(0.25, 0.55)
        cpu_spike = rng.uniform(0.30, 0.55)
        cpu_mean = _clamp(cpu_base + rng.uniform(-0.05, 0.05))
        mem_mean = rng.uniform(0.25, 0.60)
        load_mean = rng.uniform(1.0, 3.5)
        rows.append({
            "cpu_mean":       cpu_mean,
            "cpu_max":        _clamp(cpu_mean + cpu_spike),
            "cpu_std":        rng.uniform(0.12, 0.30),   # high variance is the signal
            "cpu_spike_rate": rng.uniform(0.20, 0.65),
            "mem_mean":       mem_mean,
            "mem_max":        _clamp(mem_mean + rng.uniform(0.02, 0.10)),
            "mem_std":        rng.uniform(0.01, 0.05),
            "load_mean":      load_mean,
            "load_max":       load_mean + rng.uniform(0.5, 3.5),
            "net_rx_mean":    rng.uniform(2e5, 15e6),
            "net_tx_mean":    rng.uniform(2e5, 15e6),
            "score":          rng.uniform(22, 50),
        })

    # ------------------------------------------------------------------
    # Scenario 7 -- Network-bound (high net I/O, moderate compute)
    # Edge case: ingress/egress saturated but CPU+mem look fine.
    # ------------------------------------------------------------------
    for _ in range(n):
        cpu = rng.uniform(0.10, 0.45)
        mem = rng.uniform(0.15, 0.55)
        load = rng.uniform(0.50, 2.0)
        rows.append({
            "cpu_mean":       cpu,
            "cpu_max":        _clamp(cpu + rng.uniform(0.03, 0.12)),
            "cpu_std":        rng.uniform(0.01, 0.06),
            "cpu_spike_rate": rng.uniform(0.0, 0.12),
            "mem_mean":       mem,
            "mem_max":        _clamp(mem + rng.uniform(0.02, 0.08)),
            "mem_std":        rng.uniform(0.005, 0.03),
            "load_mean":      load,
            "load_max":       load + rng.uniform(0.5, 2.0),
            "net_rx_mean":    rng.uniform(80e6, 500e6),   # near NIC capacity
            "net_tx_mean":    rng.uniform(80e6, 500e6),
            "score":          rng.uniform(30, 58),
        })

    # ------------------------------------------------------------------
    # Scenario 8 -- Mixed stress (high mem + moderate-high CPU together)
    # ------------------------------------------------------------------
    for _ in range(n):
        cpu = rng.uniform(0.55, 0.82)
        mem = rng.uniform(0.72, 0.93)
        load = rng.uniform(2.0, 5.0)
        rows.append({
            "cpu_mean":       cpu,
            "cpu_max":        _clamp(cpu + rng.uniform(0.05, 0.18)),
            "cpu_std":        rng.uniform(0.04, 0.12),
            "cpu_spike_rate": rng.uniform(0.15, 0.55),
            "mem_mean":       mem,
            "mem_max":        _clamp(mem + rng.uniform(0.01, 0.07)),
            "mem_std":        rng.uniform(0.01, 0.06),
            "load_mean":      load,
            "load_max":       load + rng.uniform(1.0, 3.0),
            "net_rx_mean":    rng.uniform(10e6, 80e6),
            "net_tx_mean":    rng.uniform(10e6, 80e6),
            "score":          rng.uniform(8, 35),
        })

    # ------------------------------------------------------------------
    # Scenario 9 -- Stressed / saturating (high CPU, high mem, high load)
    # ------------------------------------------------------------------
    for _ in range(n):
        cpu = rng.uniform(0.75, 0.99)
        mem = rng.uniform(0.72, 0.98)
        load = rng.uniform(3.0, 8.0)
        rows.append({
            "cpu_mean":       cpu,
            "cpu_max":        _clamp(cpu + rng.uniform(0.005, 0.10)),
            "cpu_std":        rng.uniform(0.04, 0.15),
            "cpu_spike_rate": rng.uniform(0.35, 1.0),
            "mem_mean":       mem,
            "mem_max":        _clamp(mem + rng.uniform(0.005, 0.08)),
            "mem_std":        rng.uniform(0.02, 0.10),
            "load_mean":      load,
            "load_max":       load + rng.uniform(1.0, 4.0),
            "net_rx_mean":    rng.uniform(20e6, 200e6),
            "net_tx_mean":    rng.uniform(20e6, 200e6),
            "score":          rng.uniform(4, 36),
        })

    # ------------------------------------------------------------------
    # Scenario 10 -- Fully saturated (worst-case edge: everything at max)
    # Remainder of samples go here so totals always sum to n_samples.
    # ------------------------------------------------------------------
    n_last = n_samples - 9 * n
    for _ in range(n_last):
        cpu = rng.uniform(0.92, 1.0)
        mem = rng.uniform(0.91, 1.0)
        load = rng.uniform(7.0, 16.0)
        rows.append({
            "cpu_mean":       cpu,
            "cpu_max":        _clamp(cpu + rng.uniform(0.0, 0.05)),
            "cpu_std":        rng.uniform(0.005, 0.06),   # low std -- pinned at max
            "cpu_spike_rate": rng.uniform(0.80, 1.0),
            "mem_mean":       mem,
            "mem_max":        _clamp(mem + rng.uniform(0.0, 0.04)),
            "mem_std":        rng.uniform(0.002, 0.04),
            "load_mean":      load,
            "load_max":       load + rng.uniform(0.5, 5.0),
            "net_rx_mean":    rng.uniform(150e6, 1e9),    # saturated NIC
            "net_tx_mean":    rng.uniform(150e6, 1e9),
            "score":          rng.uniform(0, 12),
        })

    df = pd.DataFrame(rows)
    # Shuffle so all scenarios are interleaved during training
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def train(df: pd.DataFrame) -> XGBRegressor:
    X = df[FEATURE_COLUMNS]
    y = df["score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.80,
        min_child_weight=3,
        reg_alpha=0.05,     # L1 regularisation -- helps with sparse features
        reg_lambda=1.5,     # L2 regularisation
        objective="reg:squarederror",
        eval_metric="mae",
        early_stopping_rounds=30,
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
    logger.info("Training complete -- test MAE: %.2f / 100", mae)

    # Log per-scenario performance breakdown for visibility
    df_test = X_test.copy()
    df_test["y_true"] = y_test.values
    df_test["y_pred"] = preds
    _log_score_band_mae(df_test)

    return model


def _log_score_band_mae(df_test: pd.DataFrame) -> None:
    """Log MAE broken down by the original score bands for diagnostic clarity."""
    bands = [
        ("fully_saturated",  0,   12),
        ("stressed",         4,   36),
        ("mixed_stress",     8,   35),
        ("memory_pressure",  18,  42),
        ("cpu_burst",        22,  50),
        ("network_bound",    30,  58),
        ("near_capacity",    42,  62),
        ("moderate",         40,  70),
        ("healthy_active",   78,  92),
        ("healthy_idle",     88, 100),
    ]
    for label, lo, hi in bands:
        mask = (df_test["y_true"] >= lo) & (df_test["y_true"] <= hi)
        subset = df_test[mask]
        if len(subset) > 0:
            band_mae = mean_absolute_error(subset["y_true"], subset["y_pred"])
            logger.info("  [%s] n=%d  MAE=%.2f", label, len(subset), band_mae)


def save(model: XGBRegressor) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", MODEL_PATH)


if __name__ == "__main__":
    logger.info(
        "Generating synthetic training data (%d samples, 10 scenarios)...",
        N_SAMPLES_DEFAULT,
    )
    df = generate_dataset(n_samples=N_SAMPLES_DEFAULT)

    logger.info(
        "Score distribution: min=%.1f  mean=%.1f  max=%.1f  std=%.1f",
        df["score"].min(), df["score"].mean(), df["score"].max(), df["score"].std(),
    )
    logger.info("Scenario sample counts:")
    bands = [(0, 12), (4, 36), (8, 35), (18, 42), (22, 50),
             (30, 58), (40, 70), (42, 62), (78, 92), (88, 100)]
    for lo, hi in bands:
        count = ((df["score"] >= lo) & (df["score"] <= hi)).sum()
        logger.info("  score %3d-%3d: %d samples", lo, hi, count)

    logger.info("Training XGBoost model (n_estimators=400, max_depth=6)...")
    model = train(df)

    save(model)
    logger.info("Done. Run the AI service to use this model.")
