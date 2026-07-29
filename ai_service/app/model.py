"""
model.py
--------
ML scoring model for Niyojak AI Inference Engine.

Architecture:
  - Trained with XGBoost (primary) on synthetic & real node telemetry data.
  - Input: 11 statistical features from FeatureStore sliding window.
  - Output: integer score 0-100 where 100 = perfectly healthy node.
  - Falls back to a hand-tuned heuristic formula if no trained model file exists.

The model file (niyojak_model.pkl) is loaded from MODEL_PATH on startup.
Run `python train/train_model.py` to generate the model file.

Heuristic formula (6 weighted factors, all 11 features covered):
  CPU utilisation mean:      28%
  Memory utilisation mean:   20%
  CPU spike rate (>70%):     18%
  Load average mean:         12%
  CPU std deviation:         12%  — penalises bursty/flapping nodes
  Network I/O (rx+tx):       10%  — penalises NIC-saturated nodes
"""

import os
import pickle
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger("niyojak.model")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/niyojak_model.pkl")

# Feature vector order — MUST match FEATURE_COLUMNS in train_model.py exactly
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

# ---------------------------------------------------------------------------
# Heuristic normalisation constants
# ---------------------------------------------------------------------------

# Maximum expected CPU std deviation. Training burst scenario tops out at 0.30.
# Values above this are clamped to the same penalty as fully volatile.
_CPU_STD_MAX = 0.30

# Soft NIC saturation threshold per direction (bytes/sec).
# 500 MB/s represents a busy 1 Gbps link at 50% duplex utilisation.
# Combined rx+tx is normalised against 2× this value.
_NET_CAPACITY_BPS = 500e6


class NodeScorer:
    """
    Wraps an XGBoost model (or a heuristic fallback) to score K8s nodes.

    Usage:
        scorer = NodeScorer()
        scorer.load()
        score = scorer.predict(features_dict)  # returns int 0-100
    """

    def __init__(self):
        self._model = None
        self._source = "heuristic"   # "xgboost" or "heuristic"

    def load(self):
        """Try to load the trained XGBoost model. Falls back to heuristic if file absent."""
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self._model = pickle.load(f)
                self._source = "xgboost"
                logger.info("XGBoost model loaded from %s", MODEL_PATH)
            except Exception as exc:
                logger.warning("Failed to load model from %s: %s — using heuristic", MODEL_PATH, exc)
                self._model = None
                self._source = "heuristic"
        else:
            logger.warning(
                "Model file not found at %s — using built-in heuristic scorer. "
                "Run `python train/train_model.py` to train and save a model.",
                MODEL_PATH,
            )

    def predict(self, features: dict) -> tuple[int, str]:
        """
        Score a node given its feature dict from FeatureStore.

        Returns:
            (score: int 0-100, source: str)  — score 100 = ideal placement target.
        """
        if self._model is not None:
            return self._predict_xgboost(features), "xgboost"
        return self._predict_heuristic(features), "heuristic"

    # ------------------------------------------------------------------
    # XGBoost inference
    # ------------------------------------------------------------------

    def _predict_xgboost(self, features: dict) -> int:
        vec = np.array([[features.get(col, 0.0) for col in FEATURE_COLUMNS]])
        # XGBRegressor.predict() returns a float score 0-100 (regression target)
        raw = float(self._model.predict(vec)[0])
        return int(round(max(0.0, min(100.0, raw))))

    # ------------------------------------------------------------------
    # Built-in heuristic fallback (no trained model required)
    # ------------------------------------------------------------------

    def _predict_heuristic(self, features: dict) -> int:
        """
        Six-factor scoring formula covering all signal dimensions.

        Weights:
          CPU utilisation mean:      28%  — primary placement signal
          Memory utilisation mean:   20%  — OOM pressure indicator
          CPU spike rate (>70%):     18%  — short-burst penalty
          Load average (norm/4CPU):  12%  — scheduler queue depth
          CPU std deviation:         12%  — burst/flap volatility penalty
          Network I/O (rx+tx):       10%  — NIC saturation penalty

        Each sub-score is independently clamped to [0, 1] before weighting,
        so the composite is always in [0, 1] and the returned int in [0, 100].
        A node at 0 on every metric scores 100; a fully maxed node scores 0.
        """
        cpu_score   = max(0.0, 1.0 - features.get("cpu_mean", 0.0))
        mem_score   = max(0.0, 1.0 - features.get("mem_mean", 0.0))
        spike_score = max(0.0, 1.0 - features.get("cpu_spike_rate", 0.0))

        # Load average: normalise against 4 vCPUs as a conservative baseline.
        load_norm   = min(features.get("load_mean", 0.0) / 4.0, 1.0)
        load_score  = max(0.0, 1.0 - load_norm)

        # CPU std deviation: penalises bursty/flapping nodes that look fine
        # on average but spike unpredictably (Gap 2 fix).
        std_score   = max(0.0, 1.0 - features.get("cpu_std", 0.0) / _CPU_STD_MAX)

        # Network I/O: normalise combined rx+tx against 2×NIC capacity (Gap 1 fix).
        # A fully saturated 500 MB/s duplex link drives this to 0.
        net_combined = features.get("net_rx_mean", 0.0) + features.get("net_tx_mean", 0.0)
        net_score    = max(0.0, 1.0 - net_combined / (2.0 * _NET_CAPACITY_BPS))

        composite = (
            0.28 * cpu_score   +
            0.20 * mem_score   +
            0.18 * spike_score +
            0.12 * load_score  +
            0.12 * std_score   +
            0.10 * net_score
        )
        return int(round(composite * 100))

    @property
    def source(self) -> str:
        return self._source


# Singleton instance
node_scorer = NodeScorer()
