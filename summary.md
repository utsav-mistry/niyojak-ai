# NIYOJAK — Project Summary

> **Last updated**: 2026-08-08
> **Covers**: All documentation, codebase structure, AI pipeline details, and MD file audit.

---

## 1. What is NIYOJAK?

**NIYOJAK** (Hindi: *planner/scheduler*) is an **AI-powered Kubernetes secondary scheduler**. Instead of Kubernetes' default static placement heuristics, NIYOJAK uses a trained **XGBoost machine learning model** to dynamically score nodes in real time and place pods on the healthiest available node.

Any pod setting `schedulerName: niyojak-scheduler` is intercepted by NIYOJAK instead of the default `kube-scheduler`.

---

## 2. System Architecture

```
Pod (schedulerName: niyojak-scheduler)
         |
         v
niyojak-scheduler  [Go binary — cmd/scheduler/]
  1. Watches pending pods via client-go SharedInformer
  2. Filter: taint/toleration + node affinity + resource capacity
  3. POST /score to niyojak-aiservice per eligible node
  4. Bind pod to highest-scoring node (K8s Binding API)
  5. Emit Kubernetes Event (node chosen + AI score)
         |
         v
niyojak-aiservice  [Python FastAPI — ai_service/]
  - Polls K8s API directly every 5s (node capacities + metrics-server usage + pod counts)
  - XGBoost model (18-feature contract, R²=0.9999, MAE=0.23)
  - Returns score 0–100 per (pod, node) pair in < 10ms
  - Falls back to weighted heuristic if model file is absent
```

---

## 3. Component Breakdown

| Binary | Language | Role |
|---|---|---|
| `niyojak-scheduler` | Go | From-scratch secondary scheduler: watch → filter → score → bind |
| `niyojak-aiservice` | Python / FastAPI | Inference engine: K8s API polling + XGBoost scoring |
| `niyojak-saturate` | Go | Demo tool: burns CPU/RAM on a target node |
| `niyojak-loadgen` | Go | Demo tool: HTTP flood generator to trigger HPA |
| `todo-app` | Node.js | Demo workload: To-Do app + `/admin` control portal |

---

## 4. AI Pipeline (Core Focus)

### 4.1 Offline Training — Teacher-Student Architecture

- **Teacher** (`ai_service/train/generate_dataset.py`): Pure-Python re-implementation of the Go scoring formula (`pkg/scheduler/scorer.go`). Labels every `(pod, node)` pair with a mathematically-derived placement score using a 60/40 CPU/Memory weighting and piecewise penalty functions.
- **Dataset**: 500,000 scheduling events → 23 million+ rows. Event-level train/test split to prevent leakage. Covers NaN values, resource saturation, high pod density, and thermal/network pressure edge cases.
- **Student** (`ai_service/train/train_model.py`): XGBoost Regressor learns the teacher's behavior. Output: `model/niyojak_model.json` + `model/metadata.json`.
- **Feature Contract (18 features)**: node capacities (CPU/memory), current usage, utilization %, pod count, requested resources from incoming pod, projected post-placement utilization %, headroom, balance, packing density.
- **Model Quality**: R²=0.9999, MAE=0.23, RMSE=0.29 (best iteration: 4999).

### 4.2 Real-Time Inference

- **Feature Store** (`app/feature_store.py`): Background thread polls Kubernetes API every 5s. Fetches allocatable capacity, real usage (via metrics-server), and pod count per node. Falls back to zero usage if metrics-server is down.
- **Scoring** (`app/model.py`): `NodeScorer` constructs the full 18-feature vector by combining feature-store data with the incoming pod's resource requests (projected utilization, headroom, etc.), then runs XGBoost inference.
- **Latency**: Hard `<10ms` ceiling enforced by the Go scheduler's HTTP client timeout. Any slower → fallback score of 50.

### 4.3 AI Circuit Breaker

If the AI service times out (>10ms), the Go scheduler applies score=50 (neutral / LeastAllocated). Pod scheduling is **never blocked** by the AI service.

### 4.4 Test Coverage

- `tests/test_model_accuracy.py` — deterministic unit tests with fixed seeds validating model predictions against teacher formula
- `tests/stress_test_scheduler.py` — stress tests covering heuristic scorer edge cases

---

## 5. Go Scheduler Package (`pkg/scheduler/`)

Four files implement the complete scheduling pipeline:

| File | Responsibility |
|---|---|
| `scheduler.go` | SharedInformer watch loop; coordinates filter → score → bind per pod |
| `filter.go` | `FilterNodes()`: taint/toleration, node affinity, `hasCapacity()` resource check |
| `scorer.go` | `AIScorer`: HTTP client to `/score`, 10ms timeout, `minAcceptableScore=20` threshold, heuristic fallback |
| `binder.go` | Binds pod to winning node, emits Kubernetes Event |

---

## 6. Kubernetes Infrastructure

### 6.1 3-Node Hybrid Topology

| Node | Machine | Role |
|---|---|---|
| Node 1 | Ubuntu Desktop Host | k3s Control Plane + Worker |
| Node 2 | Ubuntu Server VM (Multipass) | Worker |
| Node 3 | Ubuntu Server VM (Multipass) | Worker |

### 6.2 Observability Stack

- **Prometheus** (scrapes `node_exporter` DaemonSet) → raw hardware metrics for Grafana dashboards.
- **Grafana** (pre-configured NIYOJAK dashboard) → visual monitoring for faculty demo.
- Note: The AI feature store does **not** use Prometheus — it polls the K8s API directly.

### 6.3 Key Manifests

- `deploy/manifests/rbac.yaml` — ServiceAccount + ClusterRole + ClusterRoleBinding
- `deploy/manifests/niyojak-system.yaml` — Scheduler + AI service Deployments + ClusterIP Service

---

## 7. Demo Web App (`sample_app/`)

A full-stack **Node.js/Express To-Do App** deployed with `schedulerName: niyojak-scheduler`.

**Persistence**: JSON flat-file (`/data/tasks.json`) — no SQLite, no native bindings. Auto-created on first start.

**Two URLs:**
- `/` — Public To-Do UI (glassmorphism design, CRUD).
- `/admin` — Demo control portal.

**Admin API (key routes):**

| Route | Purpose |
|---|---|
| `GET /admin/status` | Node scores, pod map, flood status |
| `POST /admin/stress` | Launch saturate Job on a node |
| `POST /admin/release` | Delete saturate Job |
| `POST /admin/flood/start` | Begin HTTP flood |
| `POST /admin/flood/stop` | Halt flood |
| `GET /admin/events` | SSE stream — status updates every 2s |

---

## 8. Live Demo Sequence (Faculty Evaluation Flow)

1. **Healthy Baseline**: All 3 nodes ~15-20% CPU. AI scores ~85-95/100. Green gauges on `/admin`.
2. **Stress Node 2**: Click stress → Node 2 CPU spikes to 85%+, AI score drops to ~12/100.
3. **Traffic Surge**: Flood requests → HPA requests 10 new pod replicas.
4. **NIYOJAK in Action**: Scheduler queries AI → Node 1: 88, Node 2: 12, Node 3: 92 → 0 pods on Node 2.
5. **Verification**: `/admin` pod map + `kubectl get pods -o wide` live terminal proof.
6. **Recovery**: Click Release Stress → Node 2 score recovers.

---

## 9. MD File Audit — What Was Stale and What Was Fixed

| File | Status Before | Issues Found | Updated? |
|---|---|---|---|
| [README.md](README.md) | ✅ Current | None | No change needed |
| [STEP_BY_STEP_SETUP.md](STEP_BY_STEP_SETUP.md) | ✅ Current | None | No change needed |
| [CLAUDE.md](CLAUDE.md) | ⚠️ Old design doc | References `cmd/stressor/` (now `tools/saturate/`); plan is July implementation design, not current state | Left as historical record |
| [ai_service/AI_ENGINE.md](ai_service/AI_ENGINE.md) | ❌ Stale | Wrong: said Prometheus polling + sliding window. Reality: polls K8s API directly, snapshot cache. Wrong env vars (`PROMETHEUS_URL`, `WINDOW_SIZE`). Missing: `generate_dataset.py`, `tests/`, `data/`, `model/`, all 4 endpoints, real model metrics. | **Updated** |
| [cmd/BINARIES.md](cmd/BINARIES.md) | ⚠️ Slightly stale | Mentioned `stressor/` under cmd/ (doesn't exist). Missing runtime flags and nodeSelector context. | **Updated** |
| [deploy/SETUP.md](deploy/SETUP.md) | ⚠️ Incomplete | Missing manifests file list, Prometheus not used by AI (confusion point), missing access URLs table. | **Updated** |
| [pkg/PACKAGES.md](pkg/PACKAGES.md) | ❌ Stale | Called it a "kube-scheduler plugin" — it is NOT. It's a from-scratch scheduler. Only listed 1 package, 0 files. Missing `filter.go`, `scorer.go`, `binder.go`, `scheduler.go`. | **Updated** |
| [sample_app/APP.md](sample_app/APP.md) | ❌ Stale | Said SQLite + WebSocket. Reality: JSON flat-file store, SSE (not WebSocket). Listed wrong admin API details. | **Updated** |
| [tools/TOOLS.md](tools/TOOLS.md) | ✅ Accurate | Correctly describes saturate + loadgen tools and demo flow | No change needed |

---

## 10. Robustness & Production-Grade Constraints

- NaN/missing metric handling throughout the feature store.
- Early stopping in XGBoost training to prevent overfitting.
- Event-level train/test split (no row-level leakage).
- Hard 10ms latency ceiling on AI scoring calls.
- Circuit breaker: AI service down → heuristic fallback, never blocks pod scheduling.
- `minAcceptableScore=20` threshold: nodes below this are excluded unless all nodes are stressed.

---

## 11. Future Work

- Deep learning rankers (LambdaMART, XGBoost Ranker, LightGBM Ranker) for list-wise pod ranking.
- Expanding observability hooks (custom Prometheus metrics from the scheduler itself).
- SHAP-based explainability endpoint on the AI service.
- Multi-cluster / federation support.

---

*Generated from: README.md, CLAUDE.md, STEP_BY_STEP_SETUP.md, and all per-folder MD files after audit against actual source code.*
