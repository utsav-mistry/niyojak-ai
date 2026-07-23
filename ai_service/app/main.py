"""
main.py — niyojak AI Inference Engine

FastAPI server that the Go scheduler calls to score candidate nodes.
All scoring decisions are made in under 10ms using in-memory features
pre-computed by the FeatureStore background thread.

Endpoints:
  POST /score        — score a (pod, node) pair, called by niyojak-scheduler
  GET  /health       — liveness probe for Kubernetes
  GET  /nodes        — current AI scores for all known nodes (for admin dashboard)
  GET  /metrics      — Prometheus-format metrics
"""

import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from feature_store import feature_store
from model import node_scorer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("niyojak.main")

# ---------------------------------------------------------------------------
# Prometheus-style internal counters (no external library needed)
# ---------------------------------------------------------------------------

_counters = {
    "score_requests_total": 0,
    "score_fallback_total": 0,
    "score_latency_ms_sum": 0.0,
}


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("niyojak-aiservice starting")
    feature_store.start()   # launches background telemetry polling thread
    node_scorer.load()      # loads XGBoost model or falls back to heuristic
    logger.info(
        "niyojak-aiservice ready — scoring source: %s", node_scorer.source
    )
    yield
    logger.info("niyojak-aiservice shutting down")


app = FastAPI(
    title="niyojak AI Inference Engine",
    description="Real-time Kubernetes node placement scoring service",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ScoreRequest(BaseModel):
    pod_name: str
    pod_namespace: str
    node_name: str
    pod_labels: dict[str, str] = {}
    cpu_request_millicores: int = 0
    memory_request_bytes: int = 0


class ScoreResponse(BaseModel):
    score: int          # 0-100, higher is better
    reason: str
    source: str         # "xgboost" or "heuristic"


class NodeStatus(BaseModel):
    node_name: str
    score: int
    features: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    """
    Score a candidate node for a pending pod.

    Called by niyojak-scheduler (Go) for every (pod, node) pair.
    Must return within 10ms — the Go scheduler has a hard timeout.
    Features are already in memory from the FeatureStore background thread.
    """
    t0 = time.perf_counter()

    features = feature_store.get_features(req.node_name)
    raw_score, source = node_scorer.predict(features)

    # Clamp defensively
    score = max(0, min(100, raw_score))

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Update internal metrics
    _counters["score_requests_total"] += 1
    _counters["score_latency_ms_sum"] += elapsed_ms
    if source == "heuristic":
        _counters["score_fallback_total"] += 1

    reason = (
        f"cpu_mean={features.get('cpu_mean', 0):.2f} "
        f"mem_mean={features.get('mem_mean', 0):.2f} "
        f"cpu_spike_rate={features.get('cpu_spike_rate', 0):.2f} "
        f"latency={elapsed_ms:.2f}ms"
    )

    logger.debug(
        "scored node %s for pod %s/%s: %d/100 (%s) in %.2fms",
        req.node_name, req.pod_namespace, req.pod_name, score, source, elapsed_ms,
    )

    return ScoreResponse(score=score, reason=reason, source=source)


@app.get("/health")
def health():
    """Kubernetes liveness probe. Returns 200 when the service is ready."""
    return {"status": "ok", "scoring_source": node_scorer.source}


@app.get("/nodes", response_model=list[NodeStatus])
def nodes():
    """
    Return the current AI score for every known node.
    Used by the /admin dashboard to populate live gauge metrics.
    """
    result = []
    for node_name in feature_store.known_nodes():
        features = feature_store.get_features(node_name)
        raw_score, _ = node_scorer.predict(features)
        score = max(0, min(100, raw_score))
        result.append(NodeStatus(
            node_name=node_name,
            score=score,
            features={
                "cpu_mean": round(features.get("cpu_mean", 0), 3),
                "mem_mean": round(features.get("mem_mean", 0), 3),
                "cpu_spike_rate": round(features.get("cpu_spike_rate", 0), 3),
                "load_mean": round(features.get("load_mean", 0), 3),
            },
        ))
    return result


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """
    Prometheus-format metrics for scraping.
    Exposes request count, fallback count, and average scoring latency.
    """
    total = _counters["score_requests_total"]
    avg_latency = (
        _counters["score_latency_ms_sum"] / total if total > 0 else 0
    )
    lines = [
        "# HELP niyojak_score_requests_total Total scoring requests received",
        "# TYPE niyojak_score_requests_total counter",
        f"niyojak_score_requests_total {total}",
        "",
        "# HELP niyojak_score_fallback_total Requests answered by heuristic fallback",
        "# TYPE niyojak_score_fallback_total counter",
        f"niyojak_score_fallback_total {_counters['score_fallback_total']}",
        "",
        "# HELP niyojak_score_latency_ms_avg Average scoring latency in milliseconds",
        "# TYPE niyojak_score_latency_ms_avg gauge",
        f"niyojak_score_latency_ms_avg {avg_latency:.4f}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
        access_log=False,   # reduce noise; scheduler hits /score every pod
    )
