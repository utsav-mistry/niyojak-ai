#!/usr/bin/env bash
# setup_vms.sh — Provision Node 2 and Node 3 as k3s agent VMs using Multipass.
#
# What this script does:
#   1. Checks for Multipass — installs it if missing (requires snap)
#   2. Launches two Ubuntu 22.04 VMs: niyojak-node2 and niyojak-node3
#   3. Reads the k3s join token and Node 1 IP from the host
#   4. Installs k3s agent on each VM and joins it to the control plane
#   5. Waits for both nodes to become Ready
#   6. Labels nodes with their roles
#
# Usage (run on the Ubuntu Desktop host AFTER setup_host.sh):
#   chmod +x setup_vms.sh
#   ./setup_vms.sh
#
# Requirements:
#   - setup_host.sh must have been run first (k3s control plane must be up)
#   - Multipass requires KVM/QEMU. On Ubuntu Desktop this is usually available.
#
# VM specs (adjust to your hardware):
#   NODE_CPUS=2    each VM gets 2 vCPUs
#   NODE_MEM=2G    each VM gets 2 GB RAM
#   NODE_DISK=10G  each VM gets 10 GB disk

set -euo pipefail

NODE_CPUS="${NODE_CPUS:-2}"
NODE_MEM="${NODE_MEM:-2G}"
NODE_DISK="${NODE_DISK:-10G}"
VMS=("niyojak-node2" "niyojak-node3")

log()  { echo "[niyojak] $*"; }
ok()   { echo "[niyojak] OK: $*"; }
die()  { echo "[niyojak] ERROR: $*" >&2; exit 1; }

export KUBECONFIG="$HOME/.kube/config"

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

command -v kubectl >/dev/null 2>&1 || die "kubectl not found. Run setup_host.sh first."

NODE1_IP=$(hostname -I | awk '{print $1}')
JOIN_TOKEN=$(sudo cat /var/lib/rancher/k3s/server/node-token 2>/dev/null || die "k3s token not found. Run setup_host.sh first.")

log "Node 1 IP: $NODE1_IP"
log "Join token: ${JOIN_TOKEN:0:20}..."

# ---------------------------------------------------------------------------
# 1. Install Multipass if missing
# ---------------------------------------------------------------------------

if ! command -v multipass >/dev/null 2>&1; then
  log "Installing Multipass (requires snap)..."
  sudo snap install multipass
  ok "Multipass installed"
else
  ok "Multipass already installed ($(multipass version | head -1))"
fi

# ---------------------------------------------------------------------------
# 2. Launch VMs
# ---------------------------------------------------------------------------

for VM in "${VMS[@]}"; do
  if multipass info "$VM" >/dev/null 2>&1; then
    ok "VM $VM already exists — skipping launch"
  else
    log "Launching VM $VM (cpu=$NODE_CPUS mem=$NODE_MEM disk=$NODE_DISK)..."
    multipass launch 22.04 \
      --name "$VM" \
      --cpus "$NODE_CPUS" \
      --memory "$NODE_MEM" \
      --disk "$NODE_DISK"
    ok "VM $VM launched"
  fi
done

# ---------------------------------------------------------------------------
# 3. Install k3s agent on each VM and join the cluster
# ---------------------------------------------------------------------------

for VM in "${VMS[@]}"; do
  # Check if this VM is already a cluster node
  if kubectl get node "$VM" >/dev/null 2>&1; then
    ok "VM $VM is already registered in the cluster"
    continue
  fi

  VM_IP=$(multipass info "$VM" --format json | jq -r '.info."'"$VM"'".ipv4[0]')
  log "VM $VM IP: $VM_IP — installing k3s agent..."

  multipass exec "$VM" -- bash -c "
    curl -sfL https://get.k3s.io | \
      K3S_URL='https://$NODE1_IP:6443' \
      K3S_TOKEN='$JOIN_TOKEN' \
      sh -s - agent
  "

  ok "$VM joined the cluster as a worker node"
done

# ---------------------------------------------------------------------------
# 4. Wait for all nodes to be Ready
# ---------------------------------------------------------------------------

log "Waiting for all nodes to become Ready..."
for VM in "${VMS[@]}"; do
  for i in $(seq 1 30); do
    STATUS=$(kubectl get node "$VM" -o jsonpath='{.status.conditions[-1].type}' 2>/dev/null || echo "")
    if [ "$STATUS" = "Ready" ]; then
      ok "$VM is Ready"
      break
    fi
    if [ "$i" -eq 30 ]; then
      die "$VM did not become Ready within 60 seconds. Check: multipass exec $VM -- journalctl -u k3s-agent -n 50"
    fi
    sleep 2
  done
done

# ---------------------------------------------------------------------------
# 5. Label nodes for identification in the demo
# ---------------------------------------------------------------------------

kubectl label node "$(hostname)"      niyojak/role=control-plane --overwrite 2>/dev/null || true
kubectl label node "niyojak-node2"   niyojak/role=worker        --overwrite 2>/dev/null || true
kubectl label node "niyojak-node3"   niyojak/role=worker        --overwrite 2>/dev/null || true

ok "Nodes labelled with niyojak/role"

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------

echo ""
echo "----------------------------------------------------------------------"
echo "  Cluster is ready — 3 nodes"
echo "----------------------------------------------------------------------"
kubectl get nodes -o wide
echo ""
echo "  niyojak-scheduler will now distribute pods across all three nodes."
echo "  Open http://$NODE1_IP:30080/admin to watch the demo."
echo "----------------------------------------------------------------------"
