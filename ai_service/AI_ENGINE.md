# `ai_service/` — AI Inference Engine

This is the **core intelligence of NIYOJAK** — a Python FastAPI service that scores Kubernetes nodes in real time using machine learning.

## What it does

1. **Telemetry Collection** (`app/feature_store.py`): Continuously polls Prometheus (or K8s Metrics API as fallback) for node CPU, memory, network, and load metrics. Maintains a 60-second sliding window per node.

2. **ML Scoring** (`app/model.py`): Uses a trained XGBoost model (or a built-in heuristic if no model file exists) to assign each node a **placement score (0–100)**. Score 100 = perfectly healthy; Score 0 = severely contended.

3. **REST API** (`app/main.py`): Exposes a `POST /score` endpoint that the Go scheduler plugin calls per scheduling decision. Must respond in **< 10ms**.

## Directory Structure

| Path | Description |
|---|---|
| `app/main.py` | FastAPI server — entrypoint, `/score` endpoint, `/health` check, `/metrics` Prometheus exporter |
| `app/model.py` | NodeScorer class: XGBoost inference + heuristic fallback |
| `app/feature_store.py` | FeatureStore class: Prometheus/K8s Metrics polling, sliding window, feature aggregation |
| `train/train_model.py` | Offline training script — generates `niyojak_model.pkl` |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container build |

## Auto-Healing Behaviour

- **Prometheus missing**: Auto-detects, falls back to K8s Metrics API, retries Prometheus every 30s.
- **No trained model**: Falls back to heuristic scoring formula (weighted CPU + mem + load average). System works out of the box without training.

## Quick Start (local dev)

```bash
pip install -r requirements.txt
python app/main.py

# Train the model first (optional but recommended for production)
python train/train_model.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PROMETHEUS_URL` | `http://localhost:9090` | Prometheus instance URL |
| `K8S_METRICS_URL` | `https://kubernetes.default.svc` | K8s Metrics Server URL (fallback) |
| `MODEL_PATH` | `/app/model/niyojak_model.pkl` | Path to trained XGBoost model file |
| `POLL_INTERVAL_SEC` | `5` | Seconds between telemetry polls |
| `WINDOW_SIZE` | `12` | Number of readings in the sliding window (12 × 5s = 60s) |
