# Unified Plan: NIYOJAK - AI Scheduler, Self-Healing Observability & Interactive Demo App

This document presents the complete implementation plan for **NIYOJAK**, featuring an out-of-tree **AI Inference Scheduler Engine**, a **Self-Healing Auto-Bootstrapping Observability Stack**, a **3-Node Hybrid Lab Topology**, a **Full-Stack To-Do Web App with SQLite & Integrated `/admin` Control Portal**, and a **Blunt Step-by-Step Live Demo Flow for Faculty Evaluation**.

---

## 1. Blunt Step-by-Step Live Demo Execution Flow (Faculty Presentation Blueprint)

This section explains the exact step-by-step sequence of what happens during your live college project presentation:

```
+---------------------------------------------------------------------------------------------------+
|  STEP 1: HEALTHY BASELINE                                                                          |
|  - All 3 Nodes (Host Node 1, VM1 Node 2, VM2 Node 3) are healthy (~15-20% CPU).                  |
|  - Sample To-Do Web App (`/`) is running with 2 initial pod replicas.                             |
|  - Faculty sees green gauges and AI Placement Scores (~85-95/100) on `/admin`.                     |
+---------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
|  STEP 2: TRIGGER NODE STRESS                                                                      |
|  - Presenter clicks `[ ⚡ Stress Node 2 (VM1) ]` on the `/admin` portal.                            |
|  - A synthetic CPU/RAM stress container launches on Node 2.                                       |
|  - Within 2 seconds, Node 2 CPU usage spikes to 85%+.                                            |
|  - The `/admin` dashboard gauge for Node 2 turns red, and AI Score drops to 12/100 (HIGH RISK!).   |
+---------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
|  STEP 3: TRAFFIC SURGE & MULTI-POD CREATION                                                       |
|  - Presenter clicks `[ 🚀 Flood Concurrent Requests / Scale +10 Pods ]` on `/admin`.              |
|  - The backend generates 500 concurrent CRUD requests/sec to the To-Do App API.                  |
|  - Kubernetes triggers scaling and requests 10 new To-Do App pods.                                |
+---------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
|  STEP 4: NIYOJAK AI SCHEDULER IN ACTION (THE KEY FACULTY CHECKPOINT)                              |
|  - For each new pod, K8s API server invokes `niyojak-scheduler`.                                  |
|  - Go Plugin fetches real-time scores from Python AI Service (<5ms latency):                      |
|    - Node 1 (Host): Score 88/100 (HEALTHY)                                                        |
|    - Node 2 (VM1) : Score 12/100 (STRESSED - HIGH CONTENTION)                                    |
|    - Node 3 (VM2) : Score 92/100 (HEALTHY)                                                        |
|  - NIYOJAK dynamically places 0 pods on Node 2 and distributes 100% of new pods to Node 1 & 3!   |
+---------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+---------------------------------------------------------------------------------------------------+
|  STEP 5: VERIFICATION FOR FACULTY EVALUATION                                                      |
|  - Checkpoint 1 (Visual UI): `/admin` pod map animates newly created pods landing ONLY on Node 1 & 3.|
|  - Checkpoint 2 (Terminal Command): Run `kubectl get pods -o wide` live in terminal to prove      |
|    that Node 2 (`niyojak-vm1`) has ZERO new pods scheduled on it.                                 |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Self-Healing & Auto-Bootstrapping Engine

To ensure NIYOJAK runs smoothly out-of-the-box without hanging dependencies:
1. **Observability Auto-Installer**: Automatically detects if Prometheus & Grafana are missing. If missing, `setup_host.sh` automatically downloads and provisions a lightweight stack.
2. **Metrics Ingestion Fallback**: If Prometheus is warming up, telemetry falls back to direct node metrics polling (`/proc/stat` and K8s Metrics API).
3. **AI Circuit Breaker**: If AI service times out ($>10\text{ms}$), Go scheduler falls back to standard K8s `LeastAllocated` scoring without crashing pod creation.
4. **SQLite Auto-Init**: SQLite database file (`tasks.db`) and table schemas are created automatically on app startup.
5. **VM Hypervisor Setup**: `setup_vms.sh` auto-checks Multipass/KVM, provisions VM1 (Node 2) & VM2 (Node 3), and joins them to Node 1 automatically.

---

## 3. Unified Application Architecture (`sample_app/`)

```
+---------------------------------------------------------------------------------------------------+
|                            SAMPLE TO-DO WEB APP WITH INTEGRATED /ADMIN PORTAL                     |
|                                                                                                   |
|  +-------------------------------------------------+   +---------------------------------------+  |
|  | User View (`/`)                                 |   | Admin Control Portal (`/admin`)       |  |
|  | - Glassmorphic To-Do UI                         |   | - Live Node Telemetry Gauges          |  |
|  | - Task CRUD operations                          |   | - Interactive Node Flood & Release    |  |
|  | - Persistent SQLite Database (`tasks.db`)       |   | - Traffic Surge / Multi-Pod Scaling   |  |
|  |                                                 |   | - Real-time AI Pod Placement Map      |  |
|  +-------------------------------------------------+   +---------------------------------------+  |
|                                                                                                   |
|  +---------------------------------------------------------------------------------------------+  |
|  | Express REST API & Traffic Generator Backend                                                 |  |
|  | - SQLite Database Driver (`tasks.db` auto-schema)                                           |  |
|  | - K8s API Client & Traffic Burst Simulator                                                  |  |
|  +---------------------------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------------------------+
```

---

## 4. Core AI Focus & Inference Engine

```
                                  +---------------------------------------+
                                  |    Prometheus / Built-in Telemetry    |
                                  +-------------------+-------------------+
                                                      | Raw Node Metrics
                                                      v
+---------------------------------------------------------------------------------------------------+
|                                      AI INFERENCE ENGINE (FASTAPI)                                |
|                                                                                                   |
|  +------------------------------+   +-------------------------------+   +----------------------+  |
|  | Real-Time Feature Store      |   | ML Model (XGBoost/RF)         |   | Fallback Logic       |  |
|  | - 10s CPU/RAM Trend          |   | - Predicts Node Saturation    |   | - Sub-10ms response  |  |
|  | - Spike Frequency            |   | - Computes Placement Score    |   | - Least-Allocated    |  |
|  | - Thermal/Pressure Load      |   |   (0 - 100 per Node)          |   |   Safety Net         |  |
|  +------------------------------+   +-------------------------------+   +----------------------+  |
+---------------------------------------------------------------------------------------------------+
                                                      |
                                                      | Node Scores
                                                      v
                                  +---------------------------------------+
                                  |      Lean K8s Scheduler Plugin        |
                                  |    (Passes node scores to K8s)        |
                                  +---------------------------------------+
```

---

## 5. 3-Node Hybrid Topology

- **Node 1 (Control Plane + Worker)**: Ubuntu Desktop Host OS. Runs k3s/K8s control plane, Niyojak Scheduler, AI Inference Service, Telemetry Collector, and To-Do App + `/admin` pods.
- **Node 2 (Worker)**: Virtual Machine 1 (Ubuntu Server OS).
- **Node 3 (Worker)**: Virtual Machine 2 (Ubuntu Server OS).

---

## 6. Complete Project Directory Structure

```
niyojak/
├── sample_app/                  # UNIFIED FULL-STACK TO-DO APP WITH /ADMIN PORTAL
│   ├── public/                  # Frontend assets
│   │   ├── index.html           # Main To-Do App UI (/)
│   │   ├── admin.html           # Advanced Admin & Demo Portal (/admin)
│   │   ├── styles.css           # Modern Glassmorphism Styling
│   │   └── admin.js             # Live WebSocket/API caller for gauges & controls
│   ├── backend/
│   │   ├── server.js            # Node.js/Express REST API & K8s cluster controller
│   │   ├── db.js                # SQLite database driver (`tasks.db` auto-init)
│   │   ├── traffic_generator.js # Traffic surge & request simulator
│   │   └── stress_controller.js # Node flooding & release handler
│   ├── Dockerfile               # Single container build for full app
│   └── todo-app-deployment.yaml # K8s deployment spec (schedulerName: niyojak-scheduler)
├── ai_service/                  # MAIN AI INFERENCE ENGINE
│   ├── app/
│   │   ├── main.py              # FastAPI Inference Server
│   │   ├── model.py             # XGBoost/RF Model logic & Scoring
│   │   └── feature_store.py     # Real-time feature matrix calculator
│   ├── train/
│   │   └── train_model.py       # Offline model training
│   ├── requirements.txt
│   └── Dockerfile
├── cmd/
│   ├── scheduler/               # Lean Go Scheduler Plugin
│   └── stressor/                # Node Flooding binary
├── deploy/                      # 3-Node Topology Automation & Auto-Observability Setup
│   ├── setup_host.sh            # Setup Host OS as Node 1 & auto-deploy Prometheus/Grafana if missing
│   ├── setup_vms.sh             # Auto-install Multipass, provision & join VM1 (Node 2) and VM2 (Node 3)
│   └── observability/           # Lightweight fallback Prometheus & Grafana stack manifests
└── README.md
```

---

## 7. Verification Plan

1. **Live Faculty Demo Execution**: Run the 5-step flow (Healthy Baseline -> Stress Node 2 -> Surge Traffic -> Verify 0 pods on Node 2 via UI and terminal).
2. **Auto-Dependency Verification**: Run `setup_host.sh` on a clean machine to confirm automatic downloading of missing Prometheus/Grafana stack.
3. **SQLite & Web App Verification**: Test To-Do CRUD endpoints and SQLite database auto-creation.
4. **AI Scheduler Circuit Breaker Verification**: Kill AI service process -> verify Go scheduler plugin seamlessly falls back to standard K8s scheduling without crashing pod creation.
