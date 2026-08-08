# `ai_service/` — AI Inference Engine

This is the **core intelligence of NIYOJAK** — a Python FastAPI service that scores Kubernetes nodes in real time using machine learning.

## What it does

1. **Telemetry Collection** (`app/feature_store.py`): Polls the **Kubernetes API directly** every 5 seconds — node allocatable capacity via `/api/v1/nodes`, real-time usage via `/apis/metrics.k8s.io/v1beta1/nodes` (metrics-server), and pod counts via `/api/v1/pods`. Stores a per-node snapshot cache. Gracefully defaults to zero usage if metrics-server is unavailable.

2. **ML Scoring** (`app/model.py`): `NodeScorer` class loads a trained XGBoost model (`niyojak_model.json`) and scores each `(pod, node)` pair against the **18-feature contract**. Falls back to a heuristic formula if no model file is present.

3. **REST API** (`app/main.py`): Four endpoints:
   - `POST /score` — called by the Go scheduler per scheduling decision; hard `<10ms` response target
   - `GET /health` — liveness check; reports active scoring source (`xgboost` or `heuristic`)
   - `GET /nodes` — returns current AI score + key metrics for all known nodes (used by `/admin` portal)
   - `GET /metrics` — Prometheus-format counters: requests total, fallback total, average latency

## Directory Structure

| Path | Description |
|---|---|
| `app/main.py` | FastAPI server — entrypoint, all four endpoints, per-request latency tracking |
| `app/model.py` | `NodeScorer` class: XGBoost inference + heuristic fallback, loads from `MODEL_PATH` |
| `app/feature_store.py` | `FeatureStore` class: K8s API polling thread, node capacity/usage/pod-count cache |
| `train/generate_dataset.py` | **Offline** teacher dataset generator — labels 500K scheduling events (23M+ rows) using Go scorer formula |
| `train/train_model.py` | Offline XGBoost training script — produces `model/niyojak_model.json` + `model/metadata.json` |
| `model/niyojak_model.json` | Trained XGBoost model (~122 MB, R²=0.9999, MAE=0.23, trained with XGBoost 3.2.0) |
| `model/metadata.json` | Model metadata: feature list, MAE, RMSE, R², XGBoost version, best iteration |
| `tests/stress_test_scheduler.py` | Stress tests validating heuristic scorer formula and edge-case behaviour |
| `tests/test_model_accuracy.py` | Deterministic unit tests validating model predictions against teacher formula |
| `data/` | Generated training datasets (CSV + Parquet) — gitignored |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container build |

## Auto-Healing Behaviour

- **metrics-server unavailable**: CPU/memory usage defaults to 0 (node assumed idle); node remains scoreable.
- **No model file**: Falls back to heuristic scoring (weighted CPU + memory + projected utilization). No re-training required to run the demo.
- **AI timeout (>10ms)**: Enforced by the Go scheduler's HTTP client; fallback score of 50 (neutral) is applied. Pod scheduling is **never blocked**.

## Quick Start (local dev)

```bash
pip install -r requirements.txt
python app/main.py

# Train the model first (optional — heuristic fallback works without it)
python train/generate_dataset.py --target-events 500000
python train/train_model.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `K8S_API_URL` | `https://kubernetes.default.svc` | Kubernetes API server base URL |
| `MODEL_PATH` | `/app/model/niyojak_model.json` | Path to trained XGBoost model file |
| `POLL_INTERVAL_SEC` | `5` | Seconds between K8s API polls |
| `PORT` | `8000` | Port the FastAPI server listens on |

## Trained Model Metrics (from `model/metadata.json`)

| Metric | Value |
|---|---|
| R² | 0.9999 |
| MAE | 0.23 |
| RMSE | 0.29 |
| Best iteration | 4999 |
| XGBoost version | 3.2.0 |
