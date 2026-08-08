"""
feature_store.py
----------------
Real-time capacity and allocation cache for Niyojak AI Inference Engine.

Responsibility:
  - Poll Kubernetes API directly for node capacities, current usages, and pod counts.
  - Compute current_cpu_percent and current_memory_percent safely.
  - Provide a clean 18-feature compliant state for the inference engine.
"""

import os
import time
import threading
import logging
from typing import Optional

import urllib3
import requests

# Disable insecure request warnings when talking to in-cluster K8s API
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger("niyojak.feature_store")

K8S_API_URL = os.getenv("K8S_API_URL", "https://kubernetes.default.svc")
K8S_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL_SEC", "5"))


class FeatureStore:
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
        self._k8s_token: Optional[str] = None
        self._running = False

    def start(self):
        """Start the background polling thread."""
        self._load_k8s_token()
        self._running = True
        t = threading.Thread(target=self._poll_loop, daemon=True, name="niyojak-feature-store")
        t.start()
        logger.info("FeatureStore started — polling K8s API for capacity and allocation.")

    def get_features(self, node_name: str) -> dict:
        """Return the cached features for a node. Returns safe defaults if not found."""
        with self._lock:
            return self._cache.get(node_name, {
                "cluster_size": 1,
                "node_allocatable_cpu": 1000.0,
                "node_allocatable_memory": 1024.0 * 1024 * 1024,
                "current_cpu_usage": 0.0,
                "current_memory_usage": 0.0,
                "current_cpu_percent": 0.0,
                "current_memory_percent": 0.0,
                "current_pod_count": 0,
            }).copy()

    def known_nodes(self) -> list[str]:
        with self._lock:
            return list(self._cache.keys())

    def _load_k8s_token(self):
        try:
            with open(K8S_TOKEN_PATH) as f:
                self._k8s_token = f.read().strip()
        except FileNotFoundError:
            logger.warning("No K8s service account token found — API calls will be unauthenticated.")

    def _poll_loop(self):
        while self._running:
            try:
                self._poll_k8s_state()
            except Exception as exc:
                logger.warning("FeatureStore K8s poll error: %s", exc)
            time.sleep(POLL_INTERVAL)

    def _poll_k8s_state(self):
        headers = {}
        if self._k8s_token:
            headers["Authorization"] = f"Bearer {self._k8s_token}"

        # 1. Fetch Node Capacities
        r_nodes = requests.get(f"{K8S_API_URL}/api/v1/nodes", headers=headers, verify=False, timeout=4)
        r_nodes.raise_for_status()
        nodes_data = r_nodes.json().get("items", [])
        cluster_size = len(nodes_data)

        caps = {}
        for node in nodes_data:
            name = node["metadata"]["name"]
            alloc = node["status"]["allocatable"]
            caps[name] = {
                "cpu": self._parse_cpu(alloc.get("cpu", "0")),
                "memory": self._parse_memory(alloc.get("memory", "0"))
            }

        # 2. Fetch Node Metrics (Usage)
        usages = {}
        try:
            r_metrics = requests.get(f"{K8S_API_URL}/apis/metrics.k8s.io/v1beta1/nodes", headers=headers, verify=False, timeout=4)
            r_metrics.raise_for_status()
            metrics_data = r_metrics.json().get("items", [])
            for item in metrics_data:
                name = item["metadata"]["name"]
                usage = item.get("usage", {})
                usages[name] = {
                    "cpu": self._parse_cpu(usage.get("cpu", "0")),
                    "memory": self._parse_memory(usage.get("memory", "0"))
                }
        except Exception as exc:
            logger.debug("Failed to fetch node metrics (metrics-server might be down): %s", exc)

        # 3. Fetch Pod Counts
        r_pods = requests.get(f"{K8S_API_URL}/api/v1/pods", headers=headers, verify=False, timeout=4)
        r_pods.raise_for_status()
        pods_data = r_pods.json().get("items", [])
        
        pod_counts = {}
        for pod in pods_data:
            # Only count scheduled pods
            node_name = pod.get("spec", {}).get("nodeName")
            if node_name:
                pod_counts[node_name] = pod_counts.get(node_name, 0) + 1

        # 4. Atomically update cache
        new_cache = {}
        for node_name, cap in caps.items():
            alloc_cpu = cap["cpu"]
            alloc_mem = cap["memory"]
            curr_cpu = usages.get(node_name, {}).get("cpu", 0.0)
            curr_mem = usages.get(node_name, {}).get("memory", 0.0)
            
            # Safe divisions
            cpu_pct = curr_cpu / max(alloc_cpu, 1.0)
            mem_pct = curr_mem / max(alloc_mem, 1.0)
            
            new_cache[node_name] = {
                "cluster_size": cluster_size,
                "node_allocatable_cpu": alloc_cpu,
                "node_allocatable_memory": alloc_mem,
                "current_cpu_usage": curr_cpu,
                "current_memory_usage": curr_mem,
                "current_cpu_percent": cpu_pct,
                "current_memory_percent": mem_pct,
                "current_pod_count": pod_counts.get(node_name, 0),
            }
            
        with self._lock:
            self._cache = new_cache

    @staticmethod
    def _parse_cpu(cpu_str: str) -> float:
        """Parse CPU strings like '450m', '2', or '100000n' into millicores."""
        try:
            if cpu_str.endswith("m"):
                return float(cpu_str[:-1])
            if cpu_str.endswith("n"):
                return float(cpu_str[:-1]) / 1_000_000.0
            return float(cpu_str) * 1000.0
        except ValueError:
            return 0.0

    @staticmethod
    def _parse_memory(mem_str: str) -> float:
        """Parse memory strings like '8192Ki', '2Gi' into bytes."""
        try:
            if mem_str.endswith("Ki"):
                return float(mem_str[:-2]) * 1024
            if mem_str.endswith("Mi"):
                return float(mem_str[:-2]) * 1024**2
            if mem_str.endswith("Gi"):
                return float(mem_str[:-2]) * 1024**3
            if mem_str.endswith("Ti"):
                return float(mem_str[:-2]) * 1024**4
            return float(mem_str)
        except ValueError:
            return 0.0

# Singleton instance
feature_store = FeatureStore()
