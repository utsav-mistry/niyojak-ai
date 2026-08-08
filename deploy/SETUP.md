# `deploy/` — Cluster Provisioning & Observability Setup

This directory contains all scripts and manifests needed to stand up the **3-node NIYOJAK lab environment** from scratch.

## 3-Node Topology

| Role | Machine | Script |
|---|---|---|
| Node 1 — Control Plane + Worker | Ubuntu Desktop Host OS | `setup_host.sh` |
| Node 2 — Worker | Virtual Machine 1 (Ubuntu Server via Multipass) | `setup_vms.sh` |
| Node 3 — Worker | Virtual Machine 2 (Ubuntu Server via Multipass) | `setup_vms.sh` |

## Scripts

### `setup_host.sh`
Sets up the Ubuntu Desktop Host as **Node 1** (k3s control plane + worker). Also:
- Installs k3s with `--disable=traefik` and sets kubeconfig permissions.
- Creates the `niyojak-system` namespace.
- Applies RBAC, scheduler, and AI service manifests from `manifests/`.
- Deploys the observability stack from `observability/` automatically.

### `setup_vms.sh`
Provisions **Node 2** and **Node 3** as VM-based worker nodes:
- Auto-installs **Multipass** (if not present) to create lightweight Ubuntu Server VMs.
- Launches `niyojak-node2` and `niyojak-node3` with configurable CPU/RAM.
- Retrieves the k3s join token from Node 1 and registers both VMs as worker nodes automatically.

## `manifests/`

Contains the Kubernetes manifests for the NIYOJAK system itself:

| File | Description |
|---|---|
| `rbac.yaml` | `ServiceAccount`, `ClusterRole` (get/list/watch pods+nodes, create bindings+events), and `ClusterRoleBinding` for the scheduler |
| `niyojak-system.yaml` | Deployments for `niyojak-scheduler` (pinned to control-plane) and `niyojak-aiservice`; includes resource limits, liveness probes, and a `ClusterIP` Service for the AI endpoint |

## `observability/`

Lightweight Kubernetes manifests for the self-hosted observability stack:

| File | Description |
|---|---|
| `node-exporter.yaml` | DaemonSet — exposes hardware metrics on every node |
| `prometheus.yaml` | Prometheus deployment — scrapes `node_exporter` on all nodes |
| `grafana.yaml` | Grafana deployment — pre-configured with a NIYOJAK dashboard |

> **Note**: The observability stack is used for human visibility (Grafana dashboards) only. The AI feature store does **not** poll Prometheus — it polls the Kubernetes API directly.

## Usage

```bash
# On the Ubuntu Desktop host:
chmod +x setup_host.sh setup_vms.sh
./setup_host.sh        # Sets up Node 1, deploys all manifests and observability stack
./setup_vms.sh         # Provisions Node 2 and Node 3 VMs and joins them to the cluster

# Verify cluster is healthy:
kubectl get nodes -o wide
kubectl get pods -n niyojak-system
```

Expected node output:
```
NAME              STATUS   ROLES                  AGE
ubuntu-desktop    Ready    control-plane,worker   5m
niyojak-vm1       Ready    worker                 3m
niyojak-vm2       Ready    worker                 3m
```

## Access URLs (after setup)

| Service | URL |
|---|---|
| To-Do App | `http://NODE1_IP:30080` |
| Admin Portal | `http://NODE1_IP:30080/admin` |
| Prometheus | `http://NODE1_IP:30090` |
| Grafana | `http://NODE1_IP:30091` (admin / niyojak) |
