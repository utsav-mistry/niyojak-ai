# `sample_app/` — Unified To-Do Web Application

This is the **demo workload application** for NIYOJAK. It serves two purposes:

1. **`/` — Public To-Do App**: A full-stack to-do list application (frontend + REST API + SQLite database) that gets deployed as multiple replicas across the Kubernetes cluster. It generates real HTTP traffic so the cluster actually has something to schedule.

2. **`/admin` — Demo Control Portal**: A restricted admin page (hidden from the public URL) that the presenter uses during the faculty demo to trigger node stress, scale pods, and observe the AI scheduler in action — all from one browser tab.

## Directory Structure

| Path | Description |
|---|---|
| `public/index.html` | Glassmorphic To-Do UI served at `/` |
| `public/admin.html` | Admin & control portal served at `/admin` |
| `public/styles.css` | Shared dark-mode glassmorphism stylesheet |
| `public/admin.js` | WebSocket/polling client for live gauges and control buttons |
| `backend/server.js` | Node.js/Express server: To-Do CRUD routes + admin control endpoints + K8s API client |
| `backend/db.js` | SQLite driver with auto-schema initialisation (creates `tasks.db` on first run) |
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

## Admin Portal Capabilities

| Control | What it does |
|---|---|
| Stress Node N | Deploys niyojak-saturate Job on the target node, pushing its CPU to ~85% |
| Release Stress | Deletes the saturate Job, node CPU returns to idle |
| Flood Requests | Fires concurrent HTTP requests at the To-Do API; HPA detects CPU pressure and scales pods automatically |
| Stop Flood | Stops the request flood; HPA scales pods back down as load subsides |
| Live Gauges | Real-time CPU%, Memory%, AI Score for each node (WebSocket updates every 2s) |
| Pod Map | Live view of which node each running pod is scheduled on |

## SQLite Auto-Init

`backend/db.js` checks for `tasks.db` on every startup. If the file doesn't exist, it creates the database and runs the schema migration automatically. **No manual database setup is needed.**
