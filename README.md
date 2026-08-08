# NIYOJAK — AI-Powered Kubernetes Scheduler

NIYOJAK is a from-scratch Kubernetes secondary scheduler that uses a trained
XGBoost model to make pod placement decisions based on real-time node telemetry
(CPU utilisation, memory pressure, pod density).

It runs alongside the default kube-scheduler. Any pod that sets
`schedulerName: niyojak-scheduler` is scheduled by the AI engine instead of
the Kubernetes default.

---

## Architecture

```
Pod (schedulerName: niyojak-scheduler)
         |
         | Kubernetes API watch (client-go SharedInformer)
         v
niyojak-scheduler (Go binary)
   |
   |-- 1. List Ready nodes from K8s API
   |-- 2. Filter: taint/toleration + node affinity + resource capacity check
   |-- 3. POST /score to niyojak-aiservice for each eligible node
   |-- 4. Bind pod to highest-scoring node via K8s Binding API
   |-- 5. Emit Kubernetes Event (node chosen + AI score)
         |
         v
niyojak-aiservice (Python FastAPI)
   |
   |-- Polls K8s API directly every 5s:
   |       /api/v1/nodes              → allocatable capacity
   |       /apis/metrics.k8s.io/...  → real-time CPU/memory usage
   |       /api/v1/pods               → pod count per node
   |-- XGBoost model trained on 500,000 synthetic events (23M+ rows, 18-feature contract)
   |-- Returns score 0–100 per (pod, node) pair in under 10ms
   |-- Falls back to weighted heuristic if model file absent
   |-- Circuit breaker: Go scheduler enforces 10ms timeout; fallback score=50
```

---

## Components

| Binary | Language | Description |
|---|---|---|
| `niyojak-scheduler` | Go | Custom secondary scheduler — watch, filter, score, bind |
| `niyojak-aiservice` | Python | FastAPI inference engine with XGBoost + K8s API polling |
| `niyojak-saturate`  | Go | Stress tool — burns CPU and holds RAM on a target node |
| `niyojak-loadgen`   | Go | HTTP flood generator — drives HPA-based pod scaling |
| `todo-app`          | Node.js | Demo workload — To-Do app with live `/admin` control portal |

---

## Repository Layout

```
niyojak-ai/
├── cmd/scheduler/          niyojak-scheduler binary entrypoint
├── pkg/scheduler/          Scheduling logic (filter, score, bind, watch loop)
│   ├── scheduler.go        SharedInformer watch loop
│   ├── filter.go           Taint/affinity/capacity filtering
│   ├── scorer.go           AI HTTP client + heuristic fallback (10ms timeout)
│   └── binder.go           K8s Binding API + Event emission
├── ai_service/
│   ├── app/                FastAPI server, feature store, inference model
│   │   ├── main.py         /score, /health, /nodes, /metrics endpoints
│   │   ├── model.py        NodeScorer: XGBoost + heuristic fallback
│   │   └── feature_store.py  K8s API polling, per-node snapshot cache
│   ├── train/              Offline XGBoost training pipeline
│   │   ├── generate_dataset.py  Teacher dataset generator (500K events)
│   │   └── train_model.py       XGBoost training (R²=0.9999, MAE=0.23)
│   ├── model/              Trained model artifacts (gitignored — see below)
│   └── tests/              Unit and stress tests
├── tools/
│   ├── saturate/           niyojak-saturate: node CPU/RAM stress tool
│   └── loadgen/            niyojak-loadgen: HTTP load generator
├── sample_app/
│   ├── backend/            Express API + JSON store + stress/flood controllers
│   └── public/             To-Do UI and /admin control portal
└── deploy/
    ├── manifests/          RBAC + scheduler + AI service K8s manifests
    ├── observability/      Prometheus + Grafana + node-exporter
    ├── setup_host.sh       Node 1 cluster bootstrap script
    └── setup_vms.sh        Node 2 + 3 VM provisioning script
```

---

## AI Pipeline

### Teacher-Student Training (Offline)

1. **Generate dataset** (`train/generate_dataset.py`): Re-implements the Go scoring formula in Python. Labels 500,000 scheduling events (23M+ rows) — event-level split prevents data leakage. Exports to CSV/Parquet.
2. **Train model** (`train/train_model.py`): XGBoost Regressor learns the teacher's behavior across 18 features. Outputs `model/niyojak_model.json` + `model/metadata.json`.

> **Note**: The training dataset (2.5 GB Parquet) is hosted on Kaggle, not in this repo.
> Model weights are also not committed — download from Kaggle or retrain locally.

### 18-Feature Contract

Node capacity (CPU/memory allocatable), current usage, utilization %, pod count, incoming pod resource requests, projected post-placement utilization %, headroom, resource balance, packing density.

### Trained Model Quality

| Metric | Value |
|---|---|
| R² | 0.9999 |
| MAE | 0.23 |
| RMSE | 0.29 |
| Training events | 500,000 |
| Total rows | 23M+ |

---

## Demo Flow

1. Run `deploy/setup_host.sh` on the Ubuntu Desktop (Node 1).
2. Run `deploy/setup_vms.sh` to provision Node 2 and Node 3 as Multipass VMs.
3. Open `http://NODE1_IP:30080/admin` — the live admin portal.
4. Click **Stress Node 2** — burns CPU on Node 2. AI score drops to ~12/100.
5. Click **Flood Requests** — fires concurrent HTTP requests at the To-Do app.
6. HPA detects CPU > 50% and requests new pod replicas.
7. `niyojak-scheduler` places each new pod on the highest-scoring node.
8. Observe pod placement map on the admin portal — new pods land on Node 1 or Node 3 only.
9. Click **Release Stress** and watch Node 2 score recover.

---

## Quick Start (Development)

### Prerequisites

- Go 1.21+
- Python 3.11+
- Node.js 18+

### Build Go binaries

```bash
go mod download
go build -o bin/niyojak-scheduler ./cmd/scheduler/
go build -o bin/niyojak-saturate ./tools/saturate/
go build -o bin/niyojak-loadgen ./tools/loadgen/
```

> Build outputs go into `bin/` — never commit them.

### Train the AI model

```bash
cd ai_service
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download dataset from Kaggle first, place in ai_service/data/
python train/train_model.py
# or specify path: python train/train_model.py --dataset-path data/teacher_dataset.parquet
```

### Run the AI service locally

```bash
cd ai_service/app
python main.py
# POST http://localhost:8000/score
# GET  http://localhost:8000/health
# GET  http://localhost:8000/nodes
# GET  http://localhost:8000/metrics
```

### Run the To-Do App locally

```bash
cd sample_app
npm install
npm start
# http://localhost:3000
# http://localhost:3000/admin
```

---

## Cluster Setup

See `STEP_BY_STEP_SETUP.md` for the full walkthrough, or use the automation scripts:

```bash
# On Node 1 (Ubuntu Desktop host)
chmod +x deploy/setup_host.sh
./deploy/setup_host.sh

# After Node 1 is ready
chmod +x deploy/setup_vms.sh
./deploy/setup_vms.sh
```

---

## Access URLs (after cluster setup)

| Service | URL | Notes |
|---|---|---|
| To-Do App | `http://NODE1_IP:30080` | Public workload |
| Admin Portal | `http://NODE1_IP:30080/admin` | Demo control panel |
| Prometheus | `http://NODE1_IP:30090` | Raw metrics |
| Grafana | `http://NODE1_IP:30091` | Dashboards — `admin/niyojak` |
| AI Service | `http://NODE1_IP:30081` | `/health`, `/nodes`, `/metrics` |

---

## Scheduler RBAC

The scheduler requires a ClusterRole with:
- `get/list/watch` on `pods` and `nodes`
- `create` on `pods/binding`
- `create/patch` on `events`

See `deploy/manifests/rbac.yaml`.

---

## Technical Notes

- The scheduler is a **from-scratch out-of-tree secondary scheduler**, not a kube-scheduler plugin.
  It uses only `client-go` and runs as a regular pod in `niyojak-system` namespace.
- Compatible with both k3s and standard Kubernetes (v1.26+) — no cluster-side patches needed.
- The AI service has a hard 10ms timeout on scoring calls; falls back to score=50
  (neutral) so the scheduler never blocks waiting for the model.
- Node filtering (taint/toleration + node affinity + resource capacity) runs before AI scoring,
  so the model only sees nodes the pod is actually eligible to run on.
- Persistence in the demo app uses a JSON flat-file (`/data/tasks.json`), not SQLite —
  no native database bindings required.

---

## Model & Dataset

The trained model and training dataset are **not stored in this repository** due to size:

| Artifact | Location | Size |
|---|---|---|
| `teacher_dataset.parquet` | [Kaggle Dataset](https://www.kaggle.com/datasets/utsavmistry30/kaggle-kubernetes-scheduler-dataset) | ~2.5 GB |
| `niyojak_model.json` | Download from Kaggle / retrain locally | ~117 MB |

To use a pre-trained model: place `niyojak_model.json` in `ai_service/model/` before building the Docker image.
