#!/usr/bin/env bash
# setup_host.sh — Set up Node 1 (Ubuntu Desktop host) as the k3s control plane.
#
# What this script does:
#   1. Installs k3s as the control plane + worker node
#   2. Configures kubeconfig for the current user
#   3. Installs Helm (used for metrics-server if not bundled)
#   4. Checks for Prometheus and Grafana — installs from observability/ if missing
#   5. Deploys niyojak-scheduler and niyojak-aiservice
#   6. Deploys the To-Do App with HPA
#   7. Prints access URLs and join token for VMs
#
# Usage (run on the Ubuntu Desktop host, not inside a VM):
#   chmod +x setup_host.sh
#   ./setup_host.sh
#
# Re-running this script is safe — all checks are idempotent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

log()  { echo "[niyojak] $*"; }
ok()   { echo "[niyojak] OK: $*"; }
warn() { echo "[niyojak] WARN: $*"; }
die()  { echo "[niyojak] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Prerequisites
# ---------------------------------------------------------------------------

log "Checking prerequisites..."
command -v curl  >/dev/null 2>&1 || { log "Installing curl...";  sudo apt-get install -y curl; }
command -v jq    >/dev/null 2>&1 || { log "Installing jq...";    sudo apt-get install -y jq; }
command -v git   >/dev/null 2>&1 || { log "Installing git...";   sudo apt-get install -y git; }

# ---------------------------------------------------------------------------
# 2. Install k3s (control plane)
# ---------------------------------------------------------------------------

if command -v k3s >/dev/null 2>&1; then
  ok "k3s already installed ($(k3s --version | head -1))"
else
  log "Installing k3s..."
  curl -sfL https://get.k3s.io | sh -s - \
    --disable=traefik \
    --write-kubeconfig-mode=644
  log "Waiting for k3s to be ready..."
  sleep 10
fi

# ---------------------------------------------------------------------------
# 3. Configure kubeconfig for current user
# ---------------------------------------------------------------------------

KUBECONFIG_PATH="$HOME/.kube/config"
mkdir -p "$HOME/.kube"

if [ ! -f "$KUBECONFIG_PATH" ] || ! grep -q "k3s" "$KUBECONFIG_PATH" 2>/dev/null; then
  sudo cp /etc/rancher/k3s/k3s.yaml "$KUBECONFIG_PATH"
  sudo chown "$USER:$USER" "$KUBECONFIG_PATH"
  ok "kubeconfig written to $KUBECONFIG_PATH"
else
  ok "kubeconfig already configured"
fi

export KUBECONFIG="$KUBECONFIG_PATH"

# ---------------------------------------------------------------------------
# 4. Wait for the node to be Ready
# ---------------------------------------------------------------------------

log "Waiting for Node 1 to become Ready..."
for i in $(seq 1 30); do
  STATUS=$(kubectl get node "$(hostname)" -o jsonpath='{.status.conditions[-1].type}' 2>/dev/null || echo "")
  if [ "$STATUS" = "Ready" ]; then
    ok "Node 1 ($(hostname)) is Ready"
    break
  fi
  if [ "$i" -eq 30 ]; then
    die "Node 1 did not become Ready within 60 seconds. Check: journalctl -u k3s -n 50"
  fi
  sleep 2
done

# ---------------------------------------------------------------------------
# 5. Install Helm (needed for metrics-server if not already present)
# ---------------------------------------------------------------------------

if ! command -v helm >/dev/null 2>&1; then
  log "Installing Helm..."
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
  ok "Helm installed"
else
  ok "Helm already installed ($(helm version --short))"
fi

# ---------------------------------------------------------------------------
# 6. Ensure metrics-server is running (required for HPA)
# ---------------------------------------------------------------------------

if kubectl get deployment metrics-server -n kube-system >/dev/null 2>&1; then
  ok "metrics-server already running"
else
  log "Deploying metrics-server (required for HPA)..."
  kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
  # k3s sometimes needs the insecure-tls arg for metrics-server
  kubectl patch deployment metrics-server -n kube-system --type=json \
    -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]' \
    2>/dev/null || true
  ok "metrics-server deployed"
fi

# ---------------------------------------------------------------------------
# 7. Deploy observability stack (Prometheus + Grafana + node-exporter)
#    Only if not already present.
# ---------------------------------------------------------------------------

if kubectl get deployment prometheus -n niyojak-system >/dev/null 2>&1; then
  ok "Prometheus already running — skipping observability install"
else
  log "Deploying observability stack (Prometheus + Grafana + node-exporter)..."
  kubectl apply -f "$SCRIPT_DIR/observability/node-exporter.yaml"
  kubectl apply -f "$SCRIPT_DIR/observability/prometheus.yaml"
  kubectl apply -f "$SCRIPT_DIR/observability/grafana.yaml"
  ok "Observability stack deployed"
fi

# ---------------------------------------------------------------------------
# 8. Deploy niyojak-scheduler and niyojak-aiservice
# ---------------------------------------------------------------------------

log "Deploying niyojak system components..."
kubectl apply -f "$SCRIPT_DIR/manifests/rbac.yaml"
kubectl apply -f "$SCRIPT_DIR/manifests/niyojak-system.yaml"
ok "niyojak-scheduler and niyojak-aiservice deployed to niyojak-system namespace"

# ---------------------------------------------------------------------------
# 9. Deploy the To-Do sample app
# ---------------------------------------------------------------------------

log "Deploying To-Do App with HPA..."
kubectl apply -f "$REPO_ROOT/sample_app/todo-app-deployment.yaml"
ok "To-Do App deployed to default namespace"

# ---------------------------------------------------------------------------
# 10. Print join token and access URLs
# ---------------------------------------------------------------------------

NODE1_IP=$(hostname -I | awk '{print $1}')
JOIN_TOKEN=$(sudo cat /var/lib/rancher/k3s/server/node-token)

echo ""
echo "----------------------------------------------------------------------"
echo "  NIYOJAK cluster (Node 1) is ready"
echo "----------------------------------------------------------------------"
echo ""
echo "  To-Do App:         http://$NODE1_IP:30080"
echo "  Admin Portal:      http://$NODE1_IP:30080/admin"
echo "  Prometheus:        http://$NODE1_IP:30090"
echo "  Grafana:           http://$NODE1_IP:30091  (admin / niyojak)"
echo ""
echo "  Node 1 IP:         $NODE1_IP"
echo "  K3s Join Token:    $JOIN_TOKEN"
echo ""
echo "  Now run setup_vms.sh to provision Node 2 and Node 3."
echo "----------------------------------------------------------------------"
