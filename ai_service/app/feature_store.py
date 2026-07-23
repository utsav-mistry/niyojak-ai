"""
feature_store.py
----------------
Real-time telemetry feature store for Niyojak AI Inference Engine.

Responsibilities:
  - Continuously poll Prometheus for node metrics every POLL_INTERVAL_SEC seconds.
  - If Prometheus is unavailable, fall back to direct K8s Metrics API polling.
  - Maintain a sliding window of the last WINDOW_SIZE readings per node.
  - Expose `get_features(node_name)` → dict used directly by model.py for inference.

Auto-bootstrapping:
  - On startup, probes the Prometheus URL.
  - If Prometheus is not reachable, logs a warning and switches to K8s Metrics Server mode.
  - Retries Prometheus every 30 seconds in the background — auto-recovers if it comes up late.
"""

import os
import time
import threading
import logging
from collections import deque
from typing import Optional

import requests

logger = logging.getLogger("niyojak.feature_store")

# ----- Configuration (override via env vars) -----
PROMETHEUS_URL   = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
K8S_METRICS_URL  = os.getenv("K8S_METRICS_URL", "https://kubernetes.default.svc")
K8S_TOKEN_PATH   = "/var/run/secrets/kubernetes.io/serviceaccount/token"
POLL_INTERVAL    = float(os.getenv("POLL_INTERVAL_SEC", "5"))
WINDOW_SIZE      = int(os.getenv("WINDOW_SIZE", "12"))   # 12 × 5s = 60-second window
PROMETHEUS_RETRY = 30  # seconds between Prometheus availability retries


class NodeMetricsWindow:
    """Holds a fixed-size sliding window of metric readings for one node."""

    def __init__(self, window_size: int):
        self.cpu_util    = deque(maxlen=window_size)  # 0.0 – 1.0
        self.mem_util    = deque(maxlen=window_size)  # 0.0 – 1.0
        self.net_rx_bps  = deque(maxlen=window_size)  # bytes/sec received
        self.net_tx_bps  = deque(maxlen=window_size)  # bytes/sec transmitted
        self.load_avg_1m = deque(maxlen=window_size)  # 1-minute load average

    def push(self, cpu: float, mem: float, net_rx: float, net_tx: float, load: float):
        self.cpu_util.append(cpu)
        self.mem_util.append(mem)
        self.net_rx_bps.append(net_rx)
        self.net_tx_bps.append(net_tx)
        self.load_avg_1m.append(load)

    def to_feature_dict(self) -> dict:
        """
        Return statistical aggregates used as ML model input features.
        Column order MUST match FEATURE_COLUMNS in train_model.py and model.py.
        """
        def _stats(q):
            if not q:
                return {"mean": 0.0, "max": 0.0, "std": 0.0}
            vals = list(q)
            mean = sum(vals) / len(vals)
            mx   = max(vals)
            std  = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            return {"mean": mean, "max": mx, "std": std}

        cpu = _stats(self.cpu_util)
        mem = _stats(self.mem_util)
        load = _stats(self.load_avg_1m)

        return {
            # --- CPU features ---
            "cpu_mean":       cpu["mean"],
            "cpu_max":        cpu["max"],
            "cpu_std":        cpu["std"],
            # Spike indicator: fraction of readings where CPU > 70%
            "cpu_spike_rate": sum(1 for v in self.cpu_util if v > 0.70) / max(len(self.cpu_util), 1),
            # --- Memory features ---
            "mem_mean":       mem["mean"],
            "mem_max":        mem["max"],
            "mem_std":        mem["std"],
            # --- Load average ---
            "load_mean":      load["mean"],
            "load_max":       load["max"],
            # --- Network ---
            "net_rx_mean":    _stats(self.net_rx_bps)["mean"],
            "net_tx_mean":    _stats(self.net_tx_bps)["mean"],
        }


class FeatureStore:
    """
    Thread-safe, auto-healing telemetry feature store.

    Usage:
        store = FeatureStore()
        store.start()            # launches background polling thread
        feats = store.get_features("node-1")
    """

    def __init__(self):
        self._windows: dict[str, NodeMetricsWindow] = {}
        self._lock = threading.Lock()
        self._use_prometheus = False
        self._k8s_token: Optional[str] = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start the background polling thread."""
        self._probe_prometheus()
        self._load_k8s_token()
        self._running = True
        t = threading.Thread(target=self._poll_loop, daemon=True, name="niyojak-feature-store")
        t.start()
        logger.info(
            "FeatureStore started — source: %s",
            "Prometheus" if self._use_prometheus else "K8s Metrics API"
        )

    def get_features(self, node_name: str) -> dict:
        """Return feature dict for node_name. Returns zeros if node not yet seen."""
        with self._lock:
            if node_name not in self._windows:
                return NodeMetricsWindow(WINDOW_SIZE).to_feature_dict()
            return self._windows[node_name].to_feature_dict()

    def known_nodes(self) -> list[str]:
        with self._lock:
            return list(self._windows.keys())

    # ------------------------------------------------------------------
    # Internal polling loop
    # ------------------------------------------------------------------

    def _poll_loop(self):
        last_prometheus_retry = 0.0

        while self._running:
            # Periodically retry Prometheus if we're in fallback mode
            if not self._use_prometheus:
                now = time.time()
                if now - last_prometheus_retry > PROMETHEUS_RETRY:
                    self._probe_prometheus()
                    last_prometheus_retry = now

            try:
                if self._use_prometheus:
                    self._poll_prometheus()
                else:
                    self._poll_k8s_metrics()
            except Exception as exc:
                logger.warning("FeatureStore poll error: %s", exc)

            time.sleep(POLL_INTERVAL)

    # ------------------------------------------------------------------
    # Prometheus data source
    # ------------------------------------------------------------------

    def _probe_prometheus(self):
        """Check if Prometheus is reachable and switch mode accordingly."""
        try:
            r = requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=2)
            if r.status_code == 200:
                self._use_prometheus = True
                logger.info("Prometheus is available at %s", PROMETHEUS_URL)
                return
        except Exception:
            pass
        self._use_prometheus = False
        logger.warning(
            "Prometheus not reachable at %s — using K8s Metrics API fallback. "
            "Will retry every %ds.", PROMETHEUS_URL, PROMETHEUS_RETRY
        )

    def _prom_query(self, query: str) -> dict:
        """Run an instant PromQL query. Returns {node_name: float_value}.

        Priority for node identity:
          1. `node` label  — set by kube-state-metrics / node-exporter relabeling
          2. `instance` label stripped of the `:port` suffix — last resort
          3. Skip the result entirely if neither is usable
        """
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=4,
        )
        r.raise_for_status()
        result = {}
        for item in r.json().get("data", {}).get("result", []):
            metric = item["metric"]
            node = metric.get("node") or ""
            if not node:
                # Fall back to instance label but strip the `:port` suffix
                # so we get the bare IP. We still prefer the node label.
                instance = metric.get("instance", "")
                node = instance.split(":")[0] if ":" in instance else instance
            if not node or node == "unknown":
                # Skip completely — don't create a phantom entry
                continue
            result[node] = float(item["value"][1])
        return result

    def _poll_prometheus(self):
        # All queries aggregate by(node) so every result uses the canonical
        # Kubernetes node name (e.g. worker-1) not the raw instance IP:port.
        cpu_util = self._prom_query(
            '1 - avg by(node)(rate(node_cpu_seconds_total{mode="idle"}[1m]))'
        )
        mem_util = self._prom_query(
            '1 - avg by(node)(node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)'
        )
        net_rx = self._prom_query(
            'sum by(node)(rate(node_network_receive_bytes_total{device!="lo"}[1m]))'
        )
        net_tx = self._prom_query(
            'sum by(node)(rate(node_network_transmit_bytes_total{device!="lo"}[1m]))'
        )
        load_avg = self._prom_query(
            'avg by(node)(node_load1)'
        )

        all_nodes = set(cpu_util) | set(mem_util)
        with self._lock:
            # Prune any stale entries keyed by raw IP (from before this fix)
            # so they don't persist as phantom nodes in the dashboard.
            stale = [
                k for k in self._windows
                if k not in all_nodes and (k[0].isdigit() or k == "unknown")
            ]
            for k in stale:
                logger.info("FeatureStore: pruning stale node entry '%s'", k)
                del self._windows[k]

            for node in all_nodes:
                if node not in self._windows:
                    self._windows[node] = NodeMetricsWindow(WINDOW_SIZE)
                self._windows[node].push(
                    cpu=cpu_util.get(node, 0.0),
                    mem=mem_util.get(node, 0.0),
                    net_rx=net_rx.get(node, 0.0),
                    net_tx=net_tx.get(node, 0.0),
                    load=load_avg.get(node, 0.0),
                )

    # ------------------------------------------------------------------
    # K8s Metrics API fallback data source
    # ------------------------------------------------------------------

    def _load_k8s_token(self):
        try:
            with open(K8S_TOKEN_PATH) as f:
                self._k8s_token = f.read().strip()
        except FileNotFoundError:
            logger.warning("No K8s service account token found — metrics API will be unauthenticated.")

    def _poll_k8s_metrics(self):
        """Poll K8s Metrics Server `/apis/metrics.k8s.io/v1beta1/nodes` as fallback."""
        headers = {}
        if self._k8s_token:
            headers["Authorization"] = f"Bearer {self._k8s_token}"

        r = requests.get(
            f"{K8S_METRICS_URL}/apis/metrics.k8s.io/v1beta1/nodes",
            headers=headers,
            verify=False,   # self-signed cert in cluster
            timeout=4,
        )
        r.raise_for_status()
        items = r.json().get("items", [])

        with self._lock:
            for item in items:
                node = item["metadata"]["name"]
                usage = item.get("usage", {})

                # Parse CPU: "450m" → 0.45 (rough utilization; no capacity info in this API)
                cpu_str = usage.get("cpu", "0")
                cpu_cores = self._parse_cpu(cpu_str)

                # Parse memory: "1234567890" (bytes) → fraction (rough)
                mem_bytes = int(usage.get("memory", "0").rstrip("Ki")) * 1024

                if node not in self._windows:
                    self._windows[node] = NodeMetricsWindow(WINDOW_SIZE)
                # We push raw cores as cpu proxy (no total capacity from this endpoint)
                self._windows[node].push(
                    cpu=min(cpu_cores / 4.0, 1.0),   # assume 4 vCPU max for normalization
                    mem=min(mem_bytes / (8 * 1024**3), 1.0),  # assume 8GB max
                    net_rx=0.0,
                    net_tx=0.0,
                    load=0.0,
                )

    @staticmethod
    def _parse_cpu(cpu_str: str) -> float:
        """Convert K8s CPU string like '450m' or '2' to float cores."""
        if cpu_str.endswith("m"):
            return float(cpu_str[:-1]) / 1000.0
        return float(cpu_str)


# Singleton instance — imported and shared across main.py and model.py
feature_store = FeatureStore()
