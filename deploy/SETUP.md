# `deploy/` — Cluster Provisioning & Observability Setup

This directory contains all scripts and manifests needed to stand up the **3-node NIYOJAK lab environment** from scratch.

## 3-Node Topology

| Role | Machine | Script |
|---|---|---|
| Node 1 — Control Plane + Worker | Ubuntu Desktop Host OS | `setup_host.sh` |
| Node 2 — Worker | Virtual Machine 1 (Ubuntu Server) | `setup_vms.sh` |
| Node 3 — Worker | Virtual Machine 2 (Ubuntu Server) | `setup_vms.sh` |

## Scripts

### `setup_host.sh`
Sets up the Ubuntu Desktop Host as **Node 1** (k3s control plane + worker). Also:
- Installs k3s if not present.
- Deploys the Niyojak scheduler, AI Inference Engine, and To-Do App into the cluster.
- **Auto-detects Prometheus & Grafana** — installs the lightweight observability stack from `observability/` if they are missing.

### `setup_vms.sh`
Provisions **Node 2** and **Node 3** as VM-based worker nodes:
- Auto-installs **Multipass** (if not present) to create lightweight Ubuntu Server VMs.
- Launches both VMs with configurable CPU/RAM allocation.
- Retrieves the k3s join token from Node 1 and registers both VMs as worker nodes automatically.

## `observability/`

Contains lightweight Kubernetes manifests for the self-hosted observability stack:
- **Prometheus** (`prometheus.yaml`) — scrapes `node_exporter` on all nodes.
- **Grafana** (`grafana.yaml`) — pre-configured with a Niyojak dashboard.
- **node_exporter** (`node-exporter.yaml`) — DaemonSet that exposes hardware metrics on every node.

These are only deployed if Prometheus/Grafana are **not already present** in the cluster.

## Usage

```bash
# On the Ubuntu Desktop host:
chmod +x setup_host.sh setup_vms.sh
./setup_host.sh        # Sets up Node 1 and observability stack
./setup_vms.sh         # Provisions Node 2 and Node 3 VMs

# Verify cluster is healthy:
kubectl get nodes -o wide
```

Expected output:
```
NAME              STATUS   ROLES                  AGE
ubuntu-desktop    Ready    control-plane,worker   5m
niyojak-vm1       Ready    worker                 3m
niyojak-vm2       Ready    worker                 3m
```
