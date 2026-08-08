# `sample_app/` — Unified To-Do Web Application

This is the **demo workload application** for NIYOJAK. It serves two purposes:

1. **`/` — Public To-Do App**: A full-stack to-do list application (frontend + REST API + JSON file-backed persistence) that gets deployed as multiple replicas across the Kubernetes cluster. It generates real HTTP traffic so the cluster actually has something to schedule.

2. **`/admin` — Demo Control Portal**: A restricted admin page (hidden from the public URL) that the presenter uses during the faculty demo to trigger node stress, scale pods, and observe the AI scheduler in action — all from one browser tab.

## Directory Structure

| Path | Description |
|---|---|
| `public/index.html` | Glassmorphic To-Do UI served at `/` |
| `public/admin.html` | Admin & control portal served at `/admin` |
| `public/styles.css` | Shared dark-mode glassmorphism stylesheet |
| `public/admin.js` | SSE-based polling client for live gauges and control buttons |
| `backend/server.js` | Node.js/Express server: To-Do CRUD routes + all admin control endpoints + K8s API client |
| `backend/db.js` | Lightweight JSON flat-file store — auto-creates `/data/tasks.json` on first run (no native database bindings, no SQLite) |
| `backend/traffic_generator.js` | HTTP flood client — fires concurrent requests at the To-Do API to drive HPA autoscaling |
| `backend/stress_controller.js` | Calls `niyojak-saturate` (via K8s Job) on the chosen node — lives here as an admin API handler |
| `Dockerfile` | Multi-stage container build |
| `todo-app-deployment.yaml` | K8s Deployment spec with `schedulerName: niyojak-scheduler` |

## Quick Start (local dev)

```bash
npm install
node backend/server.js

# App:   http://localhost:3000/
# Admin: http://localhost:3000/admin
```

## Admin API Routes (from `backend/server.js`)

| Route | Method | What it does |
|---|---|---|
| `/admin/status` | GET | Returns current node scores, pod map, flood status |
| `/admin/stress` | POST | Launches `niyojak-saturate` Job on target node |
| `/admin/release` | POST | Deletes the saturate Job |
| `/admin/flood/start` | POST | Begins HTTP flood (configurable rps and concurrency) |
| `/admin/flood/stop` | POST | Halts the flood |
| `/admin/events` | GET | **SSE stream** — pushes status updates to the browser every 2s |

## Admin Portal Capabilities

| Control | What it does |
|---|---|
| Stress Node N | Deploys `niyojak-saturate` Job on the target node, pushing its CPU to ~85% |
| Release Stress | Deletes the saturate Job, node CPU returns to idle |
| Flood Requests | Fires concurrent HTTP requests at the To-Do API; HPA detects CPU pressure and scales pods automatically |
| Stop Flood | Stops the request flood; HPA scales pods back down as load subsides |
| Live Gauges | Real-time CPU%, Memory%, AI Score for each node (SSE updates every 2s) |
| Pod Map | Live view of which node each running pod is scheduled on |

## Persistence (db.js)

`backend/db.js` uses a **JSON flat file** (not SQLite) stored at `/data/tasks.json`. On every startup it checks for the file; if missing it creates the directory and an empty store automatically. Compatible with a shared Kubernetes `emptyDir` or `PersistentVolumeClaim` for state across restarts.

## Dependencies

- `express` — HTTP server
- `@kubernetes/client-node` — K8s API client for pod listing and Job creation
- No native database bindings (pure Node.js `fs` module for persistence)
