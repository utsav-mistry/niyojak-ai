"""
model.py
--------
ML scoring model for Niyojak AI Inference Engine.
Matches the exact 18-feature schema expected by the offline training pipeline.
"""

import os
import pickle
import logging
import numpy as np
from xgboost import XGBRegressor

logger = logging.getLogger("niyojak.model")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/niyojak_model.json")

# 18-Feature Schema exactly as defined in train_model.py
FEATURE_COLUMNS = [
    "cluster_size",
    "node_allocatable_cpu",
    "node_allocatable_memory",
    "current_cpu_usage",
    "current_memory_usage",
    "current_cpu_percent",
    "current_memory_percent",
    "current_pod_count",
    "requested_cpu",
    "requested_memory",
    "projected_cpu_percent",
    "projected_memory_percent",
    "cpu_headroom",
    "memory_headroom",
    "resource_balance",
    "cpu_request_ratio",
    "memory_request_ratio",
    "packing_density",
]

class NodeScorer:
    """Wraps the XGBoost model or heuristic fallback for node scoring."""
    
    def __init__(self):
        self._model = None
        self._source = "heuristic"

    def load(self):
        candidates = self._candidate_model_paths(MODEL_PATH)
        for candidate in candidates:
            if not os.path.exists(candidate):
                continue
            try:
                self._model = self._load_model_file(candidate)
                self._source = "xgboost"
                logger.info("XGBoost model loaded from %s", candidate)
                return
            except Exception as exc:
                logger.warning("Failed to load model from %s: %s", candidate, exc)

        logger.warning(
            "Model file not found at %s — using built-in heuristic fallback.",
            MODEL_PATH
        )
        self._model = None
        self._source = "heuristic"

    def _candidate_model_paths(self, model_path: str) -> list[str]:
        root, ext = os.path.splitext(model_path)
        if ext.lower() == ".json":
            return [model_path, root + ".pkl"]
        if ext.lower() == ".pkl":
            return [model_path, root + ".json"]
        return [model_path, root + ".json", root + ".pkl"]

    def _load_model_file(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            model = XGBRegressor()
            model.load_model(path)
            return model
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict(self, features: dict) -> tuple[int, str]:
        if self._model is not None:
            return self._predict_xgboost(features), "xgboost"
        return self._predict_heuristic(features), "heuristic"

    def _predict_xgboost(self, features: dict) -> int:
        # Extract features safely in exact expected order
        vec = np.array([[features.get(col, 0.0) for col in FEATURE_COLUMNS]])
        raw = float(self._model.predict(vec)[0])
        return int(round(max(0.0, min(100.0, raw))))

    def _predict_heuristic(self, features: dict) -> int:
        """
        Fallback scoring based on capacity and dynamic projections.
        """
        # Headroom (higher is better)
        cpu_hr = max(0.0, min(1.0, features.get("cpu_headroom", 0.0)))
        mem_hr = max(0.0, min(1.0, features.get("memory_headroom", 0.0)))
        
        # Penalties (lower is better, so 1 - penalty is higher)
        balance_penalty = max(0.0, min(1.0, features.get("resource_balance", 0.0)))
        density_penalty = max(0.0, min(1.0, features.get("packing_density", 0.0)))
        
        # High saturation penalty
        proj_cpu = features.get("projected_cpu_percent", 0.0)
        proj_mem = features.get("projected_memory_percent", 0.0)
        
        saturation_penalty = 0.0
        if proj_cpu > 0.9:
            saturation_penalty += (proj_cpu - 0.9) * 5
        if proj_mem > 0.9:
            saturation_penalty += (proj_mem - 0.9) * 5
        saturation_penalty = min(1.0, saturation_penalty)
        
        # Weights: 40% CPU HR, 40% Mem HR, 10% Balance, 10% Density
        base = (0.40 * cpu_hr) + (0.40 * mem_hr) + (0.10 * (1.0 - balance_penalty)) + (0.10 * (1.0 - density_penalty))
        
        score = base * (1.0 - saturation_penalty)
        return int(round(max(0.0, min(1.0, score)) * 100))

    @property
    def source(self) -> str:
        return self._source

node_scorer = NodeScorer()
