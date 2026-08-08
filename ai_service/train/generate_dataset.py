"""
generate_dataset.py - Niyojak Teacher Dataset Generator
========================================================

OFFLINE DEVELOPER TOOL — never called by the production scheduler or
inference engine.  The runtime AI service uses a frozen, pre-trained model.

Teacher-Student Architecture
-----------------------------
The Go scheduler is the teacher.  This script re-implements its scoring
formula in Python using the same constants and monotone penalty shape, then
uses it to label every (pod, node) pair.  No ML model is consulted here.

One row = one (pod, node) evaluation
--------------------------------------
Every node in the cluster — feasible and infeasible — is recorded.
Infeasible nodes carry feasible=False and a reject_reason.
Individual penalty components (cpu_penalty, memory_penalty, density_penalty)
are stored for model explainability.

Ranking IDs
-----------
event_id, cluster_id, pod_id let ranking frameworks (LightGBM Ranker,
XGBoost Ranker, CatBoost Ranker) group candidates per scheduling decision.
These IDs are METADATA only — never used as model input features.

Usage
-----
    python train/generate_dataset.py                         # 20 000 events
    python train/generate_dataset.py --target-events 500000
    python train/generate_dataset.py --target-events 500 --seed 0  # smoke test

Output
------
    data/teacher_dataset.csv       (always written)
    data/teacher_dataset.parquet   (written when fastparquet is installed)

References
----------
    pkg/scheduler/scorer.go              -> constants + penalty logic
    pkg/scheduler/filter.go             -> hasCapacity() feasibility check
    tests/stress_test_scheduler.py      -> heuristic_score_node() formula
"""

from __future__ import annotations

import argparse
import collections
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("niyojak.generate_dataset")

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "teacher_dataset.csv")
PARQUET_PATH = os.path.join(DATA_DIR, "teacher_dataset.parquet")


# ===========================================================================
# TEACHER SCORER CONSTANTS
# *** Must remain byte-for-byte equivalent to pkg/scheduler/scorer.go ***
# ===========================================================================

# CPU penalty thresholds  (cpuHighThreshold / cpuModerateThreshold)
CPU_HIGH_THRESHOLD: float = 0.90
CPU_MODERATE_THRESHOLD: float = 0.70
# CPU penalty values  (cpuHighPenalty / cpuModeratePenalty)
CPU_HIGH_PENALTY: int = 20
CPU_MODERATE_PENALTY: int = 10

# Memory penalty thresholds  (memHighThreshold / memModerateThreshold)
MEM_HIGH_THRESHOLD: float = 0.90
MEM_MODERATE_THRESHOLD: float = 0.70
# Memory penalty values  (memHighPenalty / memModeratePenalty)
MEM_HIGH_PENALTY: int = 20
MEM_MODERATE_PENALTY: int = 10

# Pod density penalty  (podDensityThreshold / podDensityPenalty)
POD_DENSITY_THRESHOLD: int = 30
POD_DENSITY_PENALTY: int = 5

# Score bounds  (minScore / maxScore)
MIN_SCORE: int = 0
MAX_SCORE: int = 100

# Headroom weights  (matches heuristic_score_node() in compare_heuristic_model.py)
CPU_HEADROOM_WEIGHT: float = 0.60
MEM_HEADROOM_WEIGHT: float = 0.40

# Piecewise-linear penalty anchors.
# These preserve monotonicity while avoiding hard threshold cliffs.
CPU_PENALTY_POINTS = (
    (0.00, 0.0),
    (0.40, 2.0),
    (0.60, 8.0),
    (1.00, float(CPU_HIGH_PENALTY)),
)
MEM_PENALTY_POINTS = (
    (0.00, 0.0),
    (0.40, 2.0),
    (0.60, 8.0),
    (1.00, float(MEM_HIGH_PENALTY)),
)
DENSITY_PENALTY_POINTS = (
    (0.00, 0.0),
    (0.20, 1.0),
    (0.60, 3.0),
    (1.00, float(POD_DENSITY_PENALTY)),
)


# ===========================================================================
# Node hardware profiles
# ===========================================================================

@dataclass
class HardwareProfile:
    name: str
    cpu_milli: int   # total CPU capacity in millicores
    mem_bytes: int   # total memory capacity in bytes

    @property
    def alloc_cpu_milli(self) -> int:
        """Allocatable CPU: ~5% reserved for system daemons."""
        return int(self.cpu_milli * 0.95)

    @property
    def alloc_mem_bytes(self) -> int:
        """Allocatable memory: ~8% reserved for system daemons."""
        return int(self.mem_bytes * 0.92)


def _gb(n: float) -> int:
    return int(n * 1024 ** 3)


def _piecewise_linear_penalty(value: float, points: tuple) -> float:
    """Evaluate a monotone piecewise-linear curve over normalized utilization."""
    if not points:
        return 0.0

    x = max(0.0, min(1.0, value))
    first_x, first_y = points[0]
    if x <= first_x:
        return float(first_y)

    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            if x1 <= x0:
                return float(y1)
            progress = (x - x0) / (x1 - x0)
            return float(y0 + progress * (y1 - y0))

    return float(points[-1][1])


HARDWARE_PROFILES: List[HardwareProfile] = [
    HardwareProfile("small",     cpu_milli=2_000,  mem_bytes=_gb(4)),
    HardwareProfile("medium",    cpu_milli=4_000,  mem_bytes=_gb(8)),
    HardwareProfile("large",     cpu_milli=8_000,  mem_bytes=_gb(16)),
    HardwareProfile("xlarge",    cpu_milli=16_000, mem_bytes=_gb(32)),
    HardwareProfile("cpu_heavy", cpu_milli=32_000, mem_bytes=_gb(16)),
    HardwareProfile("mem_heavy", cpu_milli=8_000,  mem_bytes=_gb(128)),
]

# Extended realistic hardware diversity
HARDWARE_PROFILES += [
    HardwareProfile("gpu",        cpu_milli=64_000, mem_bytes=_gb(64)),   # GPU instance
    HardwareProfile("arm_small",  cpu_milli=1_200,  mem_bytes=_gb(2)),    # ARM low-power
    HardwareProfile("storage_opt",cpu_milli=4_000,  mem_bytes=_gb(64)),   # storage-optimized
    HardwareProfile("burstable",  cpu_milli=6_000,  mem_bytes=_gb(8)),    # burstable instance
    HardwareProfile("edge",       cpu_milli=500,    mem_bytes=_gb(1)),    # constrained edge device
]
# Cluster size tiers: (min_nodes, max_nodes, sampling_weight)
CLUSTER_SIZE_TIERS = [
    (5,   10,  0.20),   # Tiny
    (10,  25,  0.30),   # Small
    (25,  50,  0.25),   # Medium
    (50,  100, 0.15),   # Large
    (100, 250, 0.10),   # XL
]

# Cluster scenarios — control warm-up utilization targets.
# Each entry is (scenario_name, sampling_weight).
# Low-utilisation scenarios (healthy_idle, sparse, very_light) are
# deliberately overweighted relative to equal-share (1/9 ≈ 11%) to
# counteract the label-skew toward 0: high-utilisation nodes dominate
# because adding any pod to a 70%+ node produces teacher_score=0.
# Without this correction the model regresses toward 0 for idle nodes.
SCENARIOS: list[tuple[str, float]] = [
    # (name,               weight)
    ("very_light",          0.18),   # new: proj util < 15%, teacher score 80-100
    ("healthy_idle",        0.17),   # proj util ~16-30%, teacher score 50-90
    ("sparse",              0.12),   # proj util ~10-25%
    ("balanced",            0.12),   # proj util ~40-60%
    ("mixed",               0.11),   # proj util 35-55%
    ("fragmented",          0.10),   # uneven CPU/mem util
    ("dense_packing",       0.08),   # high pod count, moderate util
    ("cpu_bottleneck",      0.06),   # high CPU util (teacher scores near 0)
    ("memory_bottleneck",   0.06),   # high mem util (teacher scores near 0)
]


# ===========================================================================
# Mutable node state
# ===========================================================================

@dataclass
class NodeState:
    node_id: str
    profile: HardwareProfile
    requested_cpu_milli: int = 0
    requested_mem_bytes: int = 0
    pod_count: int = 0

    @property
    def alloc_cpu(self) -> int:
        return self.profile.alloc_cpu_milli

    @property
    def alloc_mem(self) -> int:
        return self.profile.alloc_mem_bytes


# ===========================================================================
# Incoming pod
# ===========================================================================

@dataclass
class PodRequest:
    pod_id: str
    cpu_milli: int
    mem_bytes: int


# ===========================================================================
# TEACHER SCORER — pure Python translation of Go scheduler math
# No ML model is ever consulted in this function.
# ===========================================================================

def teacher_score_node(node: NodeState, pod: PodRequest) -> tuple:
    """
    Compute the teacher score for placing pod on node.

    Mirrors heuristic_score_node() in tests/compare_heuristic_model.py,
    which in turn mirrors calculateHeuristicPenalty() in scorer.go.

    Returns
    -------
    (teacher_score: float, cpu_penalty: float, memory_penalty: float,
     density_penalty: float, projected_cpu: float, projected_mem: float)
    """
    alloc_cpu = node.alloc_cpu
    alloc_mem = node.alloc_mem

    # Projected utilization ratios after scheduling the pod
    proj_cpu = (
        (node.requested_cpu_milli + pod.cpu_milli) / alloc_cpu
        if alloc_cpu > 0 else 0.0
    )
    proj_mem = (
        (node.requested_mem_bytes + pod.mem_bytes) / alloc_mem
        if alloc_mem > 0 else 0.0
    )

    # Headroom fractions (clamped to [0, 1])
    rem_cpu = max(0.0, 1.0 - proj_cpu)
    rem_mem = max(0.0, 1.0 - proj_mem)

    # Headroom-weighted base score (60/40 CPU/memory split)
    base = int(round(
        (CPU_HEADROOM_WEIGHT * rem_cpu + MEM_HEADROOM_WEIGHT * rem_mem) * 100
    ))

    # Piecewise-linear CPU and memory penalties begin ramping well before the
    # hard limit, then steepen as utilization approaches saturation.
    cpu_penalty = _piecewise_linear_penalty(proj_cpu, CPU_PENALTY_POINTS)
    mem_penalty = _piecewise_linear_penalty(proj_mem, MEM_PENALTY_POINTS)

    # Density is based on the projected pod count after this placement.
    density_util = min(1.0, (node.pod_count + 1) / 110.0)
    density_penalty = _piecewise_linear_penalty(density_util, DENSITY_PENALTY_POINTS)

    score = max(MIN_SCORE, min(MAX_SCORE,
                               float(base) - cpu_penalty - mem_penalty - density_penalty))
    return score, cpu_penalty, mem_penalty, density_penalty, proj_cpu, proj_mem


def feasibility_check(node: NodeState, pod: PodRequest) -> tuple:
    """
    Mirrors filter.go hasCapacity().

    Compares projected usage against node.alloc_* after placing the pod.

    Returns (feasible: bool, reject_reason: str)
    reject_reason in {"", "cpu", "memory", "both"}
    """
    cpu_ok = (node.requested_cpu_milli + pod.cpu_milli) <= node.alloc_cpu
    mem_ok = (node.requested_mem_bytes + pod.mem_bytes) <= node.alloc_mem

    if cpu_ok and mem_ok:
        return True, ""
    if not cpu_ok and not mem_ok:
        return False, "both"
    if not cpu_ok:
        return False, "cpu"
    return False, "memory"


# ===========================================================================
# Pod generation — log-normal distributions + K8s quantum rounding
# ===========================================================================

def _generate_pod(pod_id: str, rng: random.Random) -> PodRequest:
    """
    Sample CPU and memory from log-normal distributions.

    CPU:    median ~500m, range 50m-8000m, sigma=1.0
            quantised to nearest 50m (Kubernetes convention)
    Memory: median ~1GB,  range 64MB-32GB, sigma=1.2
            quantised to nearest 64MB (Kubernetes convention)
    """
    # Create a mixture distribution to give a heavier tail and special cases
    mix = rng.random()

    # Ultra-small sidecars (~10%): very small CPU/memory
    if mix < 0.10:
        cpu_raw = rng.uniform(10, 100)
        mem_raw = rng.uniform(16 * 1024 ** 2, 128 * 1024 ** 2)
    # CPU-heavy jobs (~10%): larger CPU relative to memory
    elif mix < 0.20:
        cpu_raw = math.exp(rng.gauss(math.log(2_000), 0.8))
        mem_raw = math.exp(rng.gauss(math.log(512 * 1024 ** 2), 1.0))
    # Memory-heavy jobs (~10%): larger memory relative to CPU
    elif mix < 0.30:
        cpu_raw = math.exp(rng.gauss(math.log(300), 1.0))
        mem_raw = math.exp(rng.gauss(math.log(8 * 1024 ** 3), 1.0))
    # Bursty / heavy tail (~10%): higher sigma
    elif mix < 0.40:
        cpu_raw = math.exp(rng.gauss(math.log(500), 1.8))
        mem_raw = math.exp(rng.gauss(math.log(1_073_741_824), 1.8))
    # Long-running services and AI jobs (~5%): very large
    elif mix < 0.45:
        cpu_raw = math.exp(rng.gauss(math.log(6_000), 0.7))
        mem_raw = math.exp(rng.gauss(math.log(16 * 1024 ** 3), 0.8))
    # Default: original log-normal
    else:
        cpu_raw = math.exp(rng.gauss(math.log(500), 1.0))
        mem_raw = math.exp(rng.gauss(math.log(1_073_741_824), 1.2))  # log(1 GB)

    # Quantize CPU to Kubernetes convention (50m granularity), clamp
    cpu_milli = max(50, int(round(max(10, min(32_000, cpu_raw)) / 50) * 50))

    # Quantize memory to 64MB blocks, clamp
    mem_block = 64 * 1024 * 1024                                  # 64 MB
    mem_bytes = int(
        round(max(mem_block, min(64 * 1024 ** 3, mem_raw)) / mem_block) * mem_block
    )

    return PodRequest(pod_id=pod_id, cpu_milli=cpu_milli, mem_bytes=mem_bytes)


# ===========================================================================
# Cluster generation and warm-up
# ===========================================================================

def _pick_cluster_size(rng: random.Random) -> int:
    weights = [t[2] for t in CLUSTER_SIZE_TIERS]
    r = rng.random() * sum(weights)
    cumulative = 0.0
    for lo, hi, w in CLUSTER_SIZE_TIERS:
        cumulative += w
        if r <= cumulative:
            return rng.randint(lo, hi)
    return CLUSTER_SIZE_TIERS[-1][1]


def _create_cluster(cluster_id: str, num_nodes: int, rng: random.Random) -> List[NodeState]:
    return [
        NodeState(
            node_id=f"{cluster_id}-n{i:03d}",
            profile=rng.choice(HARDWARE_PROFILES),
        )
        for i in range(num_nodes)
    ]


def _warm_up_cluster(nodes: List[NodeState], scenario: str, rng: random.Random) -> None:
    """
    Evolve the cluster into a realistic partially-loaded state before data
    collection begins.  Avoids the empty-cluster bias.

    Each scenario maps to Beta distribution (alpha, beta) parameters for
    CPU and memory utilization targets.  A 15% chance of partial workload
    removal simulates real cluster churn.
    """
    # (cpu_alpha, cpu_beta), (mem_alpha, mem_beta), avg_pod_cpu_fraction
    params = {
        # Very lightly loaded: proj util typically < 15% after pod placement.
        # These rows produce teacher_score 80-100, helping correct the
        # model's tendency to regress toward 0 for idle scenarios.
        "very_light":        ((1.0, 15.0), (1.0, 12.0), 0.060),
        "healthy_idle":      ((1.5,  8.0), (2.0,  6.0), 0.030),
        "cpu_bottleneck":    ((8.0,  1.5), (3.0,  5.0), 0.040),
        "memory_bottleneck": ((2.0,  6.0), (8.0,  1.5), 0.030),
        "fragmented":        ((2.0,  2.0), (2.0,  2.0), 0.040),
        "balanced":          ((4.0,  4.0), (4.0,  4.0), 0.030),
        "dense_packing":     ((3.0,  5.0), (3.0,  5.0), 0.015),
        "sparse":            ((1.0,  9.0), (1.0,  9.0), 0.050),
        "mixed":             ((2.5,  3.0), (2.5,  3.0), 0.035),
    }
    (cpu_a, cpu_b), (mem_a, mem_b), avg_pod_frac = params.get(
        scenario, ((2.5, 3.0), (2.5, 3.0), 0.035)
    )

    # scenario-driven churn probability (more realistic than fixed 15%)
    churn_map = {
        "very_light":       0.02,   # very low load, minimal churn
        "healthy_idle":     0.05,
        "sparse":           0.10,
        "balanced":         0.10,
        "mixed":            0.15,
        "fragmented":       0.20,
        "cpu_bottleneck":   0.20,
        "memory_bottleneck":0.20,
        "dense_packing":    0.35,
    }
    churn_p = churn_map.get(scenario, 0.15)

    for node in nodes:
        cpu_util = rng.betavariate(cpu_a, cpu_b)
        mem_util = rng.betavariate(mem_a, mem_b)
        node.requested_cpu_milli = int(node.alloc_cpu * cpu_util)
        node.requested_mem_bytes = int(node.alloc_mem * mem_util)

        avg_cpu_per_pod = max(50, int(node.alloc_cpu * avg_pod_frac))
        node.pod_count = max(0, int(node.requested_cpu_milli / avg_cpu_per_pod))

        # Simulate cluster churn with scenario-driven probability
        if rng.random() < churn_p:
            frac = rng.uniform(0.05, 0.5)
            node.requested_cpu_milli = int(node.requested_cpu_milli * (1.0 - frac))
            node.requested_mem_bytes = int(node.requested_mem_bytes * (1.0 - frac))
            node.pod_count = max(0, int(node.pod_count * (1.0 - frac)))

        # Occasionally inject short-lived spikes / bursts
        if rng.random() < 0.08:
            # add a transient burst on this node
            burst_cpu = int(node.alloc_cpu * rng.uniform(0.05, 0.25))
            burst_mem = int(node.alloc_mem * rng.uniform(0.05, 0.25))
            node.requested_cpu_milli = min(node.alloc_cpu, node.requested_cpu_milli + burst_cpu)
            node.requested_mem_bytes = min(node.alloc_mem, node.requested_mem_bytes + burst_mem)
            node.pod_count = node.pod_count + rng.randint(1, 5)


# ===========================================================================
# Feature extraction — one row per (pod, node) pair
# ===========================================================================

def _build_row(
    event_id: str,
    cluster_id: str,
    pod: PodRequest,
    cluster_size: int,
    node: NodeState,
    feasible: bool,
    reject_reason: str,
    t_score: int,
    cpu_penalty: int,
    mem_penalty: int,
    density_penalty: int,
    proj_cpu_pct: float,
    proj_mem_pct: float,
    best_node_id: str,
    winner_score: int,
    selected: bool,
    rank: Optional[int],
) -> dict:
    alloc_cpu = node.alloc_cpu
    alloc_mem = node.alloc_mem

    cur_cpu_pct = node.requested_cpu_milli / alloc_cpu if alloc_cpu > 0 else 0.0
    cur_mem_pct = node.requested_mem_bytes / alloc_mem if alloc_mem > 0 else 0.0

    proj_cpu_usage = node.requested_cpu_milli + pod.cpu_milli
    proj_mem_usage = node.requested_mem_bytes + pod.mem_bytes
    proj_rem_cpu   = max(0, alloc_cpu - proj_cpu_usage)
    proj_rem_mem   = max(0, alloc_mem - proj_mem_usage)
    proj_pod_count = node.pod_count + 1

    cpu_headroom = max(0.0, 1.0 - proj_cpu_pct)
    mem_headroom = max(0.0, 1.0 - proj_mem_pct)

    return {
        # ── Metadata (IDs — never model features) ───────────────────────
        "event_id":                   event_id,
        "cluster_id":                 cluster_id,
        "pod_id":                     pod.pod_id,
        "node_id":                    node.node_id,
        "best_node_id":               best_node_id,

        # ── Cluster context ──────────────────────────────────────────────
        "cluster_size":               cluster_size,

        # ── Node capacity ────────────────────────────────────────────────
        "node_cpu_capacity":          node.profile.cpu_milli,
        "node_memory_capacity":       node.profile.mem_bytes,
        "node_allocatable_cpu":       alloc_cpu,
        "node_allocatable_memory":    alloc_mem,

        # ── Current node state ───────────────────────────────────────────
        "current_cpu_usage":          node.requested_cpu_milli,
        "current_memory_usage":       node.requested_mem_bytes,
        "current_cpu_percent":        round(cur_cpu_pct, 6),
        "current_memory_percent":     round(cur_mem_pct, 6),
        "current_pod_count":          node.pod_count,

        # ── Incoming pod ─────────────────────────────────────────────────
        "requested_cpu":              pod.cpu_milli,
        "requested_memory":           pod.mem_bytes,

        # ── Projected state (after placement) ────────────────────────────
        "projected_cpu_usage":        proj_cpu_usage,
        "projected_memory_usage":     proj_mem_usage,
        "projected_cpu_percent":      round(proj_cpu_pct, 6),
        "projected_memory_percent":   round(proj_mem_pct, 6),
        "projected_remaining_cpu":    proj_rem_cpu,
        "projected_remaining_memory": proj_rem_mem,
        "projected_pod_count":        proj_pod_count,

        # ── Derived features ─────────────────────────────────────────────
        "cpu_headroom":               round(cpu_headroom, 6),
        "memory_headroom":            round(mem_headroom, 6),
        "cpu_fragmentation":          round(cur_cpu_pct * (1.0 - cpu_headroom), 6),
        "memory_fragmentation":       round(cur_mem_pct * (1.0 - mem_headroom), 6),
        "resource_balance":           round(abs(cpu_headroom - mem_headroom), 6),
        "cpu_waste":                  round(max(0.0, (alloc_cpu - proj_cpu_usage) / alloc_cpu)
                                            if alloc_cpu > 0 else 0.0, 6),
        "memory_waste":               round(max(0.0, (alloc_mem - proj_mem_usage) / alloc_mem)
                                            if alloc_mem > 0 else 0.0, 6),
        "cpu_request_ratio":          round(pod.cpu_milli / alloc_cpu if alloc_cpu > 0 else 0.0, 6),
        "memory_request_ratio":       round(pod.mem_bytes / alloc_mem if alloc_mem > 0 else 0.0, 6),
        "packing_density":            round(proj_pod_count / 110.0, 6),  # K8s max-pods = 110

        # ── Feasibility metadata ─────────────────────────────────────────
        "feasible":                   feasible,
        "reject_reason":              reject_reason,

        # ── Penalty components (for explainability) ──────────────────────
        "cpu_penalty":                cpu_penalty,
        "memory_penalty":             mem_penalty,
        "density_penalty":            density_penalty,

        # ── Teacher labels ────────────────────────────────────────────────
        "winner_score":               winner_score,
        "teacher_score":              t_score,
        "selected_by_scheduler":      int(selected),
        "teacher_rank":               rank,      # None for infeasible nodes
        "score_gap_from_best":        winner_score - t_score,
    }


# ===========================================================================
# Single scheduling event
# ===========================================================================

def _run_event(
    event_id: str,
    cluster_id: str,
    pod_id_counter: int,
    nodes: List[NodeState],
    rng: random.Random,
    update_state: bool,
) -> tuple:
    """
    Simulate one scheduling event: one pod arrival scored against all nodes.

    All nodes (feasible + infeasible) are included so the student model can
    learn feasibility boundaries.  Infeasible nodes receive teacher_score=0
    and teacher_rank=None.

    Returns (rows: List[dict], updated_pod_id_counter: int).
    Returns ([], counter) when no node can fit the pod — no ranking signal.
    """
    pod = _generate_pod(f"pod-{pod_id_counter:08d}", rng)
    pod_id_counter += 1

    # Feasibility pass (mirrors filter.go FilterNodes -> hasCapacity)
    feasibility: dict = {
        node.node_id: feasibility_check(node, pod)
        for node in nodes
    }
    feasible_nodes = [n for n in nodes if feasibility[n.node_id][0]]

    if not feasible_nodes:
        # No scheduling signal — skip but still advance counter
        return [], pod_id_counter

    # Teacher scoring: pure math, no ML
    scored: dict = {}
    penalties: dict = {}
    projections: dict = {}
    for node in feasible_nodes:
        t_score, cpu_p, mem_p, den_p, proj_cpu, proj_mem = teacher_score_node(node, pod)
        scored[node.node_id] = t_score
        penalties[node.node_id] = (cpu_p, mem_p, den_p)
        projections[node.node_id] = (proj_cpu, proj_mem)

    winner_score_val = max(scored.values())

    # Rank feasible nodes; ties broken by node_id (deterministic)
    ranked = sorted(
        feasible_nodes,
        key=lambda n: (-scored[n.node_id], n.node_id),
    )
    ranks = {n.node_id: i + 1 for i, n in enumerate(ranked)}
    winner_node = ranked[0]

    # Build rows for ALL nodes
    rows = []
    for node in nodes:
        is_feasible, reject_reason = feasibility[node.node_id]

        if is_feasible:
            t_score  = scored[node.node_id]
            cpu_p, mem_p, den_p = penalties[node.node_id]
            proj_cpu, proj_mem  = projections[node.node_id]
            rank     = ranks[node.node_id]
            selected = (node.node_id == winner_node.node_id)
        else:
            # Infeasible: score=0, penalties=0, no rank
            t_score  = 0
            cpu_p = mem_p = den_p = 0
            # Still compute projections for feature columns (informative even if infeasible)
            proj_cpu = (node.requested_cpu_milli + pod.cpu_milli) / node.alloc_cpu \
                if node.alloc_cpu > 0 else 0.0
            proj_mem = (node.requested_mem_bytes + pod.mem_bytes) / node.alloc_mem \
                if node.alloc_mem > 0 else 0.0
            rank     = None
            selected = False

        rows.append(_build_row(
            event_id=event_id,
            cluster_id=cluster_id,
            pod=pod,
            cluster_size=len(nodes),
            node=node,
            feasible=is_feasible,
            reject_reason=reject_reason,
            t_score=t_score,
            cpu_penalty=cpu_p,
            mem_penalty=mem_p,
            density_penalty=den_p,
            proj_cpu_pct=proj_cpu,
            proj_mem_pct=proj_mem,
            best_node_id=winner_node.node_id,
            winner_score=winner_score_val,
            selected=selected,
            rank=rank,
        ))

    # Evolve cluster state: winner binds the pod
    if update_state:
        winner_node.requested_cpu_milli += pod.cpu_milli
        winner_node.requested_mem_bytes += pod.mem_bytes
        winner_node.pod_count += 1

    return rows, pod_id_counter


# ===========================================================================
# Main generation loop
# ===========================================================================


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    if mins < 60:
        return f"{mins}m{secs:02d}s"
    hrs = mins // 60
    rem_mins = mins % 60
    return f"{hrs}h{rem_mins:02d}m"


def _get_ram_mb() -> Optional[float]:
    """Best-effort RAM usage for progress logs. Returns None when unavailable."""
    try:
        import importlib

        psutil = importlib.import_module("psutil")
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 ** 2)
    except Exception:
        return None


class DatasetChunkWriter:
    """Streaming writer for CSV and optional Parquet output."""

    def __init__(self, csv_path: str, parquet_path: str, write_parquet: bool) -> None:
        self.csv_path = csv_path
        self.parquet_path = parquet_path
        self.write_parquet = write_parquet
        self.csv_header_written = False
        self.parquet_initialized = False
        self.rows_written = 0
        self.chunks_written = 0
        self.parquet_available = False

        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        if os.path.exists(self.parquet_path):
            os.remove(self.parquet_path)

        if self.write_parquet:
            try:
                import fastparquet  # noqa: F401

                self.parquet_available = True
            except ImportError:
                logger.warning("fastparquet not installed — Parquet streaming disabled.")
                self.parquet_available = False

    def write_chunk(self, rows: List[dict]) -> int:
        if not rows:
            return 0

        df_chunk = pd.DataFrame(rows)

        df_chunk.to_csv(
            self.csv_path,
            mode="a",
            header=not self.csv_header_written,
            index=False,
        )
        self.csv_header_written = True

        if self.write_parquet and self.parquet_available:
            try:
                import fastparquet

                fastparquet.write(
                    self.parquet_path,
                    df_chunk,
                    compression="SNAPPY",
                    file_scheme="simple",
                    write_index=False,
                    append=self.parquet_initialized,
                )
                self.parquet_initialized = True
            except Exception as exc:
                logger.warning(
                    "Parquet chunk append failed (%s). Continuing with CSV-only stream.",
                    exc,
                )
                self.write_parquet = False

        chunk_len = len(df_chunk)
        self.rows_written += chunk_len
        self.chunks_written += 1
        return chunk_len


class RunningStat:
    """Numerically stable running min/mean/max/std estimator."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def update(self, value: float) -> None:
        self.count += 1
        if value < self.min_val:
            self.min_val = value
        if value > self.max_val:
            self.max_val = value
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def std(self) -> float:
        if self.count == 0:
            return 0.0
        return math.sqrt(self.m2 / self.count)


class StreamingValidator:
    """Streaming validation metrics without materializing the full dataset."""

    def __init__(self) -> None:
        self.total_rows = 0
        self.total_events = 0
        self.unique_clusters: Set[str] = set()
        self.total_feasible = 0
        self.total_infeasible = 0
        self.sum_cluster_size = 0.0
        self.sum_feasible_nodes_per_event = 0.0

        self.teacher_score_stat = RunningStat()
        self.score_hist = [0] * 10

        self.rank_counts = collections.Counter()
        self.penalty_stats = {
            "cpu_penalty": RunningStat(),
            "memory_penalty": RunningStat(),
            "density_penalty": RunningStat(),
        }
        self.penalty_hist = {
            "cpu_penalty": collections.Counter(),
            "memory_penalty": collections.Counter(),
            "density_penalty": collections.Counter(),
        }
        self.penalty_zero_counts = {
            "cpu_penalty": 0,
            "memory_penalty": 0,
            "density_penalty": 0,
        }
        self.penalty_unique_values = {
            "cpu_penalty": set(),
            "memory_penalty": set(),
            "density_penalty": set(),
        }

        self.cpu_request_stat = RunningStat()
        self.mem_request_stat = RunningStat()
        self.cur_cpu_pct_stat = RunningStat()
        self.cur_mem_pct_stat = RunningStat()
        self.proj_cpu_pct_feasible_stat = RunningStat()
        self.proj_mem_pct_feasible_stat = RunningStat()

        self.selected_sum = 0
        self.winner_gap_nonzero = 0
        self.winner_rank_not_one = 0
        self.feasible_rank_null = 0

    def update_event(self, rows: List[dict]) -> None:
        if not rows:
            return

        self.total_events += 1
        self.unique_clusters.add(rows[0]["cluster_id"])
        self.sum_cluster_size += float(rows[0]["cluster_size"])

        feasible_in_event = 0
        for row in rows:
            self.total_rows += 1
            feasible = bool(row["feasible"])

            if feasible:
                feasible_in_event += 1
                self.total_feasible += 1

                score = float(row["teacher_score"])
                self.teacher_score_stat.update(score)
                bin_idx = min(9, max(0, int(score // 10)))
                self.score_hist[bin_idx] += 1

                rank = row["teacher_rank"]
                if rank is None or (isinstance(rank, float) and math.isnan(rank)):
                    self.feasible_rank_null += 1
                else:
                    self.rank_counts[int(rank)] += 1

                for key in ("cpu_penalty", "memory_penalty", "density_penalty"):
                    val = float(row[key])
                    self.penalty_stats[key].update(val)
                    if abs(val) < 1e-12:
                        self.penalty_zero_counts[key] += 1
                    self.penalty_unique_values[key].add(round(val, 3))
                    self.penalty_hist[key][round(val, 1)] += 1
            else:
                self.total_infeasible += 1

            self.cpu_request_stat.update(float(row["requested_cpu"]))
            self.mem_request_stat.update(float(row["requested_memory"]))
            self.cur_cpu_pct_stat.update(float(row["current_cpu_percent"]) * 100.0)
            self.cur_mem_pct_stat.update(float(row["current_memory_percent"]) * 100.0)

            if feasible:
                self.proj_cpu_pct_feasible_stat.update(float(row["projected_cpu_percent"]) * 100.0)
                self.proj_mem_pct_feasible_stat.update(float(row["projected_memory_percent"]) * 100.0)

            selected = int(row["selected_by_scheduler"])
            self.selected_sum += selected
            if selected == 1:
                if float(row["score_gap_from_best"]) != 0.0:
                    self.winner_gap_nonzero += 1
                if row["teacher_rank"] != 1:
                    self.winner_rank_not_one += 1

        self.sum_feasible_nodes_per_event += feasible_in_event

    def print_report(self) -> None:
        sep = "=" * 72
        print(f"\n{sep}\n TEACHER DATASET VALIDATION REPORT (STREAMING)\n{sep}")

        n_total = self.total_rows
        n_events = self.total_events
        n_clusters = len(self.unique_clusters)
        n_pods = n_events
        n_feas = self.total_feasible
        n_infeas = self.total_infeasible

        print("\nDATASET OVERVIEW")
        print(f"  Total rows            : {n_total:,}")
        print(f"  Scheduling events     : {n_events:,}")
        print(f"  Unique clusters       : {n_clusters:,}")
        print(f"  Unique pods           : {n_pods:,}")
        if n_total > 0:
            print(f"  Feasible rows         : {n_feas:,}  ({100*n_feas/n_total:.1f}%)")
            print(f"  Infeasible rows       : {n_infeas:,}  ({100*n_infeas/n_total:.1f}%)")
        else:
            print("  Feasible rows         : 0  (0.0%)")
            print("  Infeasible rows       : 0  (0.0%)")

        avg_size = (self.sum_cluster_size / n_events) if n_events > 0 else 0.0
        avg_feas = (self.sum_feasible_nodes_per_event / n_events) if n_events > 0 else 0.0
        print(f"  Avg cluster size      : {avg_size:.1f} nodes")
        print(f"  Avg feasible nodes/ev : {avg_feas:.1f}")

        print("\nTEACHER SCORE DISTRIBUTION (feasible nodes)")
        if n_feas > 0:
            print(
                f"  min={self.teacher_score_stat.min_val:.0f}  "
                f"mean={self.teacher_score_stat.mean:.1f}  "
                f"max={self.teacher_score_stat.max_val:.0f}  "
                f"std={self.teacher_score_stat.std():.1f}"
            )
            mx = max(self.score_hist) or 1
            for i, count in enumerate(self.score_hist):
                lo = i * 10
                hi = (i + 1) * 10
                bar = "#" * int(count / mx * 30)
                print(f"  [{lo:3d}-{hi:3d}]: {count:8,}  {bar}")
        else:
            print("  No feasible rows.")

        print("\nTEACHER RANK DISTRIBUTION (feasible nodes, top 10)")
        if self.rank_counts:
            for rank, cnt in sorted(self.rank_counts.items())[:10]:
                print(f"  Rank {rank:3}: {cnt:8,}")
            if len(self.rank_counts) > 10:
                print(f"  ... ({len(self.rank_counts) - 10} more ranks)")
        else:
            print("  No rank data.")

        print("\nPENALTY COMPONENT DISTRIBUTION (feasible nodes)")
        for col, label in [
            ("cpu_penalty", "CPU penalty    "),
            ("memory_penalty", "Memory penalty "),
            ("density_penalty", "Density penalty"),
        ]:
            stats = self.penalty_stats[col]
            zero_pct = (100.0 * self.penalty_zero_counts[col] / n_feas) if n_feas > 0 else 0.0
            print(
                f"  {label}: min={stats.min_val:.2f}  mean={stats.mean:.2f}  "
                f"max={stats.max_val:.2f}  std={stats.std():.2f}  "
                f"zeros={zero_pct:.1f}%  unique_values={len(self.penalty_unique_values[col]):,}"
            )

        print("\nPENALTY HISTOGRAMS (rounded to 0.1)")
        for col, label, width in [
            ("cpu_penalty", "CPU penalty    ", float(CPU_HIGH_PENALTY)),
            ("memory_penalty", "Memory penalty ", float(MEM_HIGH_PENALTY)),
            ("density_penalty", "Density penalty", float(POD_DENSITY_PENALTY)),
        ]:
            hist = self.penalty_hist[col]
            if not hist:
                continue
            mx = max(hist.values()) or 1
            print(f"  {label}")
            for bucket in sorted(hist)[:12]:
                bar = "#" * int(hist[bucket] / mx * 24)
                print(f"    {bucket:>5.1f} / {width:.1f}: {hist[bucket]:8,}  {bar}")

        print("\nPOD REQUEST DISTRIBUTION")
        if self.cpu_request_stat.count > 0:
            print(
                f"  CPU  min={int(self.cpu_request_stat.min_val):,} mCPU  "
                f"mean={self.cpu_request_stat.mean:.1f} mCPU  "
                f"max={int(self.cpu_request_stat.max_val):,} mCPU"
            )
            print(
                f"  Mem  min={self.mem_request_stat.min_val/1024**3:.2f}  "
                f"mean={self.mem_request_stat.mean/1024**3:.2f}  "
                f"max={self.mem_request_stat.max_val/1024**3:.2f}  GB"
            )

        print("\nNODE UTILIZATION DISTRIBUTION (current)")
        if self.cur_cpu_pct_stat.count > 0:
            print(
                f"  CPU: min={self.cur_cpu_pct_stat.min_val:.1f}%  "
                f"mean={self.cur_cpu_pct_stat.mean:.1f}%  "
                f"max={self.cur_cpu_pct_stat.max_val:.1f}%  "
                f"std={self.cur_cpu_pct_stat.std():.1f}%"
            )
            print(
                f"  Mem: min={self.cur_mem_pct_stat.min_val:.1f}%  "
                f"mean={self.cur_mem_pct_stat.mean:.1f}%  "
                f"max={self.cur_mem_pct_stat.max_val:.1f}%  "
                f"std={self.cur_mem_pct_stat.std():.1f}%"
            )

        print("\nNODE UTILIZATION DISTRIBUTION (projected, feasible rows only)")
        if self.proj_cpu_pct_feasible_stat.count > 0:
            print(
                f"  CPU: min={self.proj_cpu_pct_feasible_stat.min_val:.1f}%  "
                f"mean={self.proj_cpu_pct_feasible_stat.mean:.1f}%  "
                f"max={self.proj_cpu_pct_feasible_stat.max_val:.1f}%  "
                f"std={self.proj_cpu_pct_feasible_stat.std():.1f}%"
            )
            print(
                f"  Mem: min={self.proj_mem_pct_feasible_stat.min_val:.1f}%  "
                f"mean={self.proj_mem_pct_feasible_stat.mean:.1f}%  "
                f"max={self.proj_mem_pct_feasible_stat.max_val:.1f}%  "
                f"std={self.proj_mem_pct_feasible_stat.std():.1f}%"
            )

        print("\nUTILIZATION HARD LIMIT CHECKS")
        cur_cpu_ok = self.cur_cpu_pct_stat.max_val <= 100.0
        cur_mem_ok = self.cur_mem_pct_stat.max_val <= 100.0
        proj_cpu_ok = self.proj_cpu_pct_feasible_stat.max_val <= 100.0
        proj_mem_ok = self.proj_mem_pct_feasible_stat.max_val <= 100.0
        print(f"  Max current CPU <= 100%     : {'PASS' if cur_cpu_ok else 'FAIL'}")
        print(f"  Max current memory <= 100%  : {'PASS' if cur_mem_ok else 'FAIL'}")
        print(f"  Max projected CPU <= 100%   : {'PASS' if proj_cpu_ok else 'FAIL'}  (feasible rows only)")
        print(f"  Max projected memory <= 100%: {'PASS' if proj_mem_ok else 'FAIL'}  (feasible rows only)")

        print("\nFEATURE DOMINANCE CHECK")
        print("  Streaming mode does not compute exact Gini without full-column materialization.")

        print("\nLABEL INTEGRITY CHECKS")
        print(f"  selected_by_scheduler sum   : {self.selected_sum:,}  (expected ~{n_events:,})")
        gap_ok = self.winner_gap_nonzero == 0
        print(f"  Gap=0 for all winners       : {'PASS' if gap_ok else 'FAIL'}")
        r1_ok = self.winner_rank_not_one == 0
        print(f"  Winners always rank 1       : {'PASS' if r1_ok else 'FAIL'}")
        no_null = self.feasible_rank_null == 0
        print(f"  No null rank (feasible)     : {'PASS' if no_null else 'FAIL'}")
        print(f"\n{sep}\n")


def analyze_generated_dataset(
    csv_path: str,
    sample_rows: int = 50_000,
    top_k: int = 10,
) -> None:
    """Optional offline analysis pass for feature redundancy checks."""
    if sample_rows <= 0:
        return
    if not os.path.exists(csv_path):
        logger.warning("Feature analysis skipped: %s not found", csv_path)
        return

    df = pd.read_csv(csv_path, nrows=sample_rows)
    if df.empty:
        logger.warning("Feature analysis skipped: dataset sample is empty")
        return

    candidate_features = [
        "cluster_size",
        "node_cpu_capacity",
        "node_memory_capacity",
        "node_allocatable_cpu",
        "node_allocatable_memory",
        "current_cpu_usage",
        "current_memory_usage",
        "current_cpu_percent",
        "current_memory_percent",
        "current_pod_count",
        "requested_cpu",
        "requested_memory",
        "projected_cpu_usage",
        "projected_memory_usage",
        "projected_cpu_percent",
        "projected_memory_percent",
        "projected_remaining_cpu",
        "projected_remaining_memory",
        "projected_pod_count",
        "cpu_headroom",
        "memory_headroom",
        "cpu_fragmentation",
        "memory_fragmentation",
        "resource_balance",
        "cpu_waste",
        "memory_waste",
        "cpu_request_ratio",
        "memory_request_ratio",
        "packing_density",
    ]

    feature_cols = [c for c in candidate_features if c in df.columns]
    numeric_df = df[feature_cols + ["teacher_score"]].select_dtypes(include=[np.number]).copy()

    print("\nFEATURE ANALYSIS (sample)")

    constant_cols = [col for col in feature_cols if numeric_df[col].nunique(dropna=False) <= 1]
    if constant_cols:
        print("  Constant features:")
        for col in constant_cols:
            print(f"    - {col}")
    else:
        print("  Constant features: none")

    correlations = []
    for col in feature_cols:
        if col == "teacher_score":
            continue
        series = numeric_df[col]
        if series.nunique(dropna=True) <= 1:
            continue
        corr = series.corr(numeric_df["teacher_score"])
        if pd.notna(corr):
            correlations.append((abs(float(corr)), float(corr), col))
    correlations.sort(reverse=True)

    print("  Top feature correlations with teacher_score:")
    for abs_corr, corr, col in correlations[:top_k]:
        print(f"    - {col:<28} corr={corr:+.4f}")

    numeric_features = [c for c in feature_cols if c in numeric_df.columns and c != "teacher_score"]
    corr_matrix = numeric_df[numeric_features].corr().abs()
    high_pairs: List[tuple] = []
    for i, a in enumerate(numeric_features):
        for b in numeric_features[i + 1:]:
            value = float(corr_matrix.loc[a, b])
            if value >= 0.95:
                high_pairs.append((value, a, b))
    high_pairs.sort(reverse=True)

    if high_pairs:
        print("  Highly correlated feature pairs (|corr| >= 0.95):")
        for value, a, b in high_pairs[:top_k]:
            print(f"    - {a} <-> {b}  corr={value:.4f}")
    else:
        print("  Highly correlated feature pairs: none >= 0.95")

def generate_dataset(
    target_events: int,
    events_per_cluster: int = 20,
    seed: int = 42,
    chunk_size: int = 100_000,
    skip_parquet: bool = False,
) -> None:
    """
    Generate the teacher dataset.

    Parameters
    ----------
    target_events : int
        Number of scheduling events to simulate.
        Total rows ~ target_events * average_cluster_size.
    events_per_cluster : int
        Pod arrivals per cluster before creating a new one.
        Higher = more realistic cluster evolution across events.
    seed : int
        Master seed for reproducibility.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    rng = random.Random(seed)
    row_buffer: List[dict] = []
    writer = DatasetChunkWriter(CSV_PATH, PARQUET_PATH, write_parquet=(not skip_parquet))
    validator = StreamingValidator()
    events_done = 0
    cluster_count = 0
    pod_id_counter = 0
    skipped = 0

    logger.info(
        "Teacher dataset generation: target_events=%d events_per_cluster=%d seed=%d chunk_size=%d",
        target_events, events_per_cluster, seed, chunk_size,
    )
    t0 = time.time()

    while events_done < target_events:
        num_nodes  = _pick_cluster_size(rng)
        cluster_id = f"c{cluster_count:06d}"
        cluster_count += 1

        nodes    = _create_cluster(cluster_id, num_nodes, rng)
        # Weighted scenario sampling — low-utilisation scenarios are overweighted
        # to counteract the natural label skew toward 0 from high-utilisation events.
        _scenario_names   = [s[0] for s in SCENARIOS]
        _scenario_weights = [s[1] for s in SCENARIOS]
        scenario = rng.choices(_scenario_names, weights=_scenario_weights, k=1)[0]
        _warm_up_cluster(nodes, scenario, rng)

        for _ in range(events_per_cluster):
            if events_done >= target_events:
                break

            event_id = f"e{events_done:08d}"
            rows, pod_id_counter = _run_event(
                event_id=event_id,
                cluster_id=cluster_id,
                pod_id_counter=pod_id_counter,
                nodes=nodes,
                rng=rng,
                update_state=True,
            )

            if rows:
                events_done += 1
                validator.update_event(rows)
                row_buffer.extend(rows)

                if len(row_buffer) >= chunk_size:
                    writer.write_chunk(row_buffer)
                    row_buffer.clear()
            else:
                skipped += 1

            if events_done > 0 and events_done % 10_000 == 0:
                elapsed = time.time() - t0
                rate    = events_done / elapsed
                eta     = (target_events - events_done) / rate if rate > 0 else 0.0
                ram_mb = _get_ram_mb()
                ram_txt = f" | RAM {ram_mb:.1f} MB" if ram_mb is not None else ""
                logger.info(
                    "Progress: %d/%d events | rows=%d | chunks=%d | clusters=%d | ETA %s%s",
                    events_done,
                    target_events,
                    writer.rows_written + len(row_buffer),
                    writer.chunks_written,
                    cluster_count,
                    _format_seconds(eta),
                    ram_txt,
                )

    if row_buffer:
        writer.write_chunk(row_buffer)
        row_buffer.clear()

    logger.info(
        "Done in %.1fs: %d events | %d rows written | %d chunks | %d clusters | %d skipped (no-fit)",
        time.time() - t0,
        events_done,
        writer.rows_written,
        writer.chunks_written,
        cluster_count,
        skipped,
    )
    validator.print_report()
    logger.info("CSV written : %s  (%.1f MB)", CSV_PATH, os.path.getsize(CSV_PATH) / 1024 ** 2)
    if writer.parquet_initialized and os.path.exists(PARQUET_PATH):
        logger.info("Parquet written : %s  (%.1f MB)", PARQUET_PATH, os.path.getsize(PARQUET_PATH) / 1024 ** 2)


# ===========================================================================
# Validation report
# ===========================================================================

def _gini(arr: np.ndarray) -> float:
    """Gini coefficient — measures feature dominance (0=uniform, 1=concentrated)."""
    arr = np.sort(np.abs(arr[np.isfinite(arr)]))
    if len(arr) == 0 or arr[-1] == 0:
        return 0.0
    n = len(arr)
    return float((n + 1 - 2.0 * np.cumsum(arr).sum() / np.cumsum(arr)[-1]) / n)


def print_validation_report(df: pd.DataFrame) -> None:
    sep = "=" * 72
    print(f"\n{sep}\n TEACHER DATASET VALIDATION REPORT\n{sep}")

    n_total    = len(df)
    n_events   = df["event_id"].nunique()
    n_clusters = df["cluster_id"].nunique()
    n_pods     = df["pod_id"].nunique()
    n_feas     = int(df["feasible"].sum())
    n_infeas   = n_total - n_feas

    print("\nDATASET OVERVIEW")
    print(f"  Total rows            : {n_total:,}")
    print(f"  Scheduling events     : {n_events:,}")
    print(f"  Unique clusters       : {n_clusters:,}")
    print(f"  Unique pods           : {n_pods:,}")
    print(f"  Feasible rows         : {n_feas:,}  ({100*n_feas/n_total:.1f}%)")
    print(f"  Infeasible rows       : {n_infeas:,}  ({100*n_infeas/n_total:.1f}%)")
    avg_size  = df.groupby("event_id")["cluster_size"].first().mean()
    feas_df   = df[df["feasible"]].copy()
    avg_feas  = feas_df.groupby("event_id").size().mean()
    print(f"  Avg cluster size      : {avg_size:.1f} nodes")
    print(f"  Avg feasible nodes/ev : {avg_feas:.1f}")

    print("\nTEACHER SCORE DISTRIBUTION (feasible nodes)")
    scores = feas_df["teacher_score"].values.astype(float)
    print(f"  min={scores.min():.0f}  mean={scores.mean():.1f}  "
          f"max={scores.max():.0f}  std={scores.std():.1f}")
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    hist, edges = np.histogram(scores, bins=bins)
    mx = hist.max() or 1
    for i in range(len(hist)):
        bar = "\u2588" * int(hist[i] / mx * 30)
        print(f"  [{edges[i]:3.0f}-{edges[i+1]:3.0f}]: {hist[i]:8,}  {bar}")

    print("\nTEACHER RANK DISTRIBUTION (feasible nodes, top 10)")
    rc = feas_df["teacher_rank"].dropna().astype(int).value_counts().sort_index()
    for rank, cnt in rc.head(10).items():
        print(f"  Rank {rank:3}: {cnt:8,}")
    if len(rc) > 10:
        print(f"  ... ({len(rc) - 10} more ranks)")

    print("\nPENALTY COMPONENT DISTRIBUTION (feasible nodes)")
    for col, label in [
        ("cpu_penalty",     "CPU penalty    "),
        ("memory_penalty",  "Memory penalty "),
        ("density_penalty", "Density penalty"),
    ]:
        v = feas_df[col].values
        zero_pct = 100.0 * (v == 0).sum() / len(v) if len(v) > 0 else 0.0
        unique_values = len({round(float(x), 3) for x in v}) if len(v) > 0 else 0
        print(
            f"  {label}: mean={v.mean():.2f}  std={v.std():.2f}  "
            f"zeros={zero_pct:.1f}%  unique_values={unique_values:,}"
        )

    print("\nPOD REQUEST DISTRIBUTION")
    print(f"  CPU  min={int(df['requested_cpu'].min()):,} mCPU  "
          f"mean={df['requested_cpu'].mean():.1f} mCPU  "
          f"max={int(df['requested_cpu'].max()):,} mCPU")
    print(f"  Mem  min={df['requested_memory'].min()/1024**3:.2f}  "
          f"mean={df['requested_memory'].mean()/1024**3:.2f}  "
          f"max={df['requested_memory'].max()/1024**3:.2f}  GB")

    print("\nNODE UTILIZATION DISTRIBUTION (current)")
    for col, lbl in [("current_cpu_percent", "CPU"), ("current_memory_percent", "Mem")]:
        v = df[col].values * 100
        print(f"  {lbl}: min={v.min():.1f}%  mean={v.mean():.1f}%  "
              f"max={v.max():.1f}%  std={v.std():.1f}%")

    numeric_features = [
        "current_cpu_percent", "current_memory_percent",
        "projected_cpu_percent", "projected_memory_percent",
        "cpu_headroom", "memory_headroom",
        "cpu_fragmentation", "memory_fragmentation",
        "resource_balance", "packing_density",
        "cpu_request_ratio", "memory_request_ratio",
    ]
    print("\nFEATURE DOMINANCE CHECK (Gini coefficient)")
    print(f"  {'Feature':<30} {'Gini':>6}")
    print(f"  {'-'*30} {'-'*6}")
    max_gini = 0.0
    for feat in numeric_features:
        if feat in df.columns:
            g = _gini(df[feat].fillna(0).values)
            max_gini = max(max_gini, g)
            flag = "  !! HIGH" if g > 0.8 else ""
            print(f"  {feat:<30} {g:6.3f}{flag}")
    status = "WARN: dominance detected" if max_gini > 0.8 else "PASS"
    print(f"\n  Dominance check: {status}")

    print("\nLABEL INTEGRITY CHECKS")
    n_sel = int(df["selected_by_scheduler"].sum())
    print(f"  selected_by_scheduler sum   : {n_sel:,}  (expected ~{n_events:,})")
    gap_ok = (feas_df[feas_df["selected_by_scheduler"] == 1]["score_gap_from_best"] == 0).all()
    print(f"  Gap=0 for all winners       : {'PASS' if gap_ok else 'FAIL'}")
    r1_ok = (feas_df[feas_df["selected_by_scheduler"] == 1]["teacher_rank"] == 1).all()
    print(f"  Winners always rank 1       : {'PASS' if r1_ok else 'FAIL'}")
    no_null = feas_df["teacher_rank"].notna().all()
    print(f"  No null rank (feasible)     : {'PASS' if no_null else 'FAIL'}")

    print(f"\n{sep}\n")


# ===========================================================================
# Output writers
# ===========================================================================

def write_outputs(df: pd.DataFrame, skip_parquet: bool = False) -> None:
    """Write CSV (always) and Parquet via fastparquet (no pyarrow)."""
    os.makedirs(DATA_DIR, exist_ok=True)

    df.to_csv(CSV_PATH, index=False)
    logger.info("CSV written : %s  (%.1f MB)", CSV_PATH,
                os.path.getsize(CSV_PATH) / 1024 ** 2)

    if skip_parquet:
        return

    try:
        import fastparquet  # noqa: F401
        df.to_parquet(PARQUET_PATH, engine="fastparquet", index=False)
        logger.info("Parquet written : %s  (%.1f MB)", PARQUET_PATH,
                    os.path.getsize(PARQUET_PATH) / 1024 ** 2)
    except ImportError:
        logger.warning("fastparquet not installed — Parquet skipped.  "
                       "pip install fastparquet")
    except Exception as exc:
        logger.warning("Parquet write failed (%s) — CSV is still valid.", exc)


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Niyojak teacher dataset for offline training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-events",      type=int, default=20_000)
    parser.add_argument("--events-per-cluster", type=int, default=20)
    parser.add_argument("--seed",               type=int, default=42)
    parser.add_argument("--chunk-size",         type=int, default=100_000)
    parser.add_argument("--no-parquet",         action="store_true")
    parser.add_argument("--analyze-features",   action="store_true")
    parser.add_argument("--analysis-sample-rows", type=int, default=50_000)
    args = parser.parse_args()

    generate_dataset(
        target_events=args.target_events,
        events_per_cluster=args.events_per_cluster,
        seed=args.seed,
        chunk_size=args.chunk_size,
        skip_parquet=args.no_parquet,
    )

    if args.analyze_features:
        analyze_generated_dataset(CSV_PATH, sample_rows=args.analysis_sample_rows)


if __name__ == "__main__":
    main()
