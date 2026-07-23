# NIYOJAK — AI-Powered Kubernetes Scheduler

NIYOJAK is a from-scratch Kubernetes secondary scheduler that uses a trained
XGBoost model to make pod placement decisions based on real-time node telemetry
(CPU utilisation, memory pressure, load average, network I/O).

It runs alongside the default kube-scheduler. Any pod that sets
`schedulerName: niyojak-scheduler` is scheduled by the AI engine instead of
the Kubernetes default.

---

## Architecture

```
Pod (schedulerName: niyojak-scheduler)
         |
         | API Server watches
         v
niyojak-scheduler (Go binary)
   |
   |-- 1. List Ready nodes from K8s API
   |-- 2. Filter: taint/toleration + resource capacity check
   |-- 3. POST /score to niyojak-aiservice for each eligible node
   |-- 4. Bind pod to highest-scoring node via K8s Binding API
   |-- 5. Emit Kubernetes Event (visible in kubectl describe pod)
         |
         v
niyojak-aiservice (Python FastAPI)
   |
   |-- Polls Prometheus every 5s (falls back to K8s Metrics API)
   |-- Maintains 60-second sliding window of node telemetry
   |-- XGBoost regressor trained on 8,000 synthetic samples
   |-- Returns score 0-100 per (pod, node) pair in under 10ms
   |-- Falls back to weighted heuristic if model file absent
```

---

## Components

| Binary | Language | Description |
|---|---|---|
| `niyojak-scheduler` | Go | Custom secondary scheduler — watch, filter, score, bind |
| `niyojak-aiservice` | Python | FastAPI inference engine with XGBoost + Prometheus |
| `niyojak-saturate`  | Go | Stress tool — burns CPU and holds RAM on a target node |
| `niyojak-loadgen`   | Go | HTTP flood generator — drives HPA-based pod scaling |
| `todo-app`          | Node.js | Demo workload — To-Do app with live admin control portal |

---

## Repository Layout

```
niyojak/
├── cmd/scheduler/          niyojak-scheduler binary entrypoint
├── pkg/scheduler/          Scheduling logic (filter, score, bind, watch loop)
├── ai_service/
│   ├── app/                FastAPI server, feature store, inference model
│   └── train/              Offline XGBoost training script
├── tools/
│   ├── saturate/           niyojak-saturate: node stress tool
│   └── loadgen/            niyojak-loadgen: HTTP load generator
├── sample_app/
│   ├── backend/            Express API + SQLite + stress/flood controllers
│   └── public/             To-Do UI and admin control portal
└── deploy/
    ├── manifests/          RBAC + scheduler + AI service K8s manifests
    ├── observability/      Prometheus + Grafana + node-exporter
    ├── setup_host.sh       Node 1 cluster bootstrap script
    └── setup_vms.sh        Node 2 + 3 VM provisioning script
```

---

## Demo Flow

1. Run `deploy/setup_host.sh` on the Ubuntu Desktop (Node 1).
2. Run `deploy/setup_vms.sh` to provision Node 2 and Node 3 as Multipass VMs.
3. Open `http://NODE1_IP:30080/admin` — the live admin portal.
4. Click **Stress Node 1** — burns CPU on Node 1 via a K8s Job.
5. Watch the AI score on Node 1 drop (updated every 2 seconds).
6. Click **Flood Requests** — fires concurrent HTTP requests at the To-Do app.
7. HPA detects CPU > 50% and requests new pod replicas.
8. `niyojak-scheduler` places each new pod on the highest-scoring node.
9. Observe pod placement map on the admin portal — new pods land on Node 2 or Node 3.
10. Click **Release Stress** and watch Node 1 score recover.

---

## Quick Start (Development)

### Prerequisites

- Go 1.21+
- Python 3.11+
- Node.js 18+

### Build Go binaries

```bash
go mod download
go build ./cmd/scheduler/
go build ./tools/saturate/
go build ./tools/loadgen/
```

### Train the AI model

```bash
cd ai_service
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train/train_model.py
```

### Run the AI service locally

```bash
cd ai_service/app
python main.py
# POST http://localhost:8000/score
# GET  http://localhost:8000/health
# GET  http://localhost:8000/nodes
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

See `deploy/SETUP.md` for detailed instructions.

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

---

## Scheduler RBAC

The scheduler requires a ClusterRole with:
- `get/list/watch` on `pods` and `nodes`
- `create` on `pods/binding`
- `create/patch` on `events`

See `deploy/manifests/rbac.yaml`.

---

## Technical Notes

- The scheduler is a **secondary custom scheduler**, not a kube-scheduler plugin.
  It uses only `client-go` and runs as a regular pod in `niyojak-system` namespace.
- Compatible with both k3s and standard Kubernetes — no cluster-side patches needed.
- The AI service has a hard 10ms timeout on scoring calls; falls back to score=50
  (neutral) so the scheduler never blocks waiting for the model.
- Node filtering (taint/toleration + resource capacity) runs before AI scoring,
  so the model only sees nodes the pod is actually eligible to run on.