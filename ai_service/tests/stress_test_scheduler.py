"""
stress_test_scheduler.py — Hardened Stress-Test Suite for Niyojak AI Scheduler
-------------------------------------------------------------------------------
Simulates an intensely constrained, high-load Kubernetes cluster with heavy pod
demands and saturated nodes to rigorously test edge-case scheduling behavior.

Penalty formula
---------------
The AI model was trained on teacher_score computed by generate_dataset.py, which
uses a PIECEWISE-LINEAR penalty that begins ramping at 40% utilisation.  This
file uses the same formula so that the AI penalty correction is faithful to what
the model learned.

A separate legacy_step_penalty() function preserves the old hard step-function
(thresholds at 70%/90%) as a clearly-labelled reference baseline.
"""

import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb

# ---------------------------------------------------------------------------
# Model path
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "niyojak_model.json")

# ---------------------------------------------------------------------------
# Feature columns — strictly matching model specs (18 features)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "cluster_size",
    "node_allocatable_cpu",
    "node_allocatable_memory",
    "current_cpu_usage",
    "current_memory_usage",
    "current_cpu_percent",
    "current_memory_percent",
    "current_pod_count",
    "requested_cpu",
    "requested_memory",
    "projected_cpu_percent",
    "projected_memory_percent",
    "cpu_headroom",
    "memory_headroom",
    "resource_balance",
    "cpu_request_ratio",
    "memory_request_ratio",
    "packing_density",
]

# ---------------------------------------------------------------------------
# Score bounds
# ---------------------------------------------------------------------------
MIN_SCORE = 0
MAX_SCORE = 100

# ---------------------------------------------------------------------------
# PIECEWISE-LINEAR penalty constants
# Must remain equivalent to generate_dataset.py so that penalty correction
# applied to AI raw scores is faithful to the training teacher formula.
# ---------------------------------------------------------------------------
CPU_PENALTY_POINTS = (
    (0.00,  0.0),
    (0.40,  2.0),
    (0.60,  8.0),
    (1.00, 20.0),
)
MEM_PENALTY_POINTS = (
    (0.00,  0.0),
    (0.40,  2.0),
    (0.60,  8.0),
    (1.00, 20.0),
)
DENSITY_PENALTY_POINTS = (
    (0.00, 0.0),
    (0.20, 1.0),
    (0.60, 3.0),
    (1.00, 5.0),
)

# ---------------------------------------------------------------------------
# Legacy step-function constants (kept for heuristic-baseline comparison)
# ---------------------------------------------------------------------------
CPU_HIGH_THRESHOLD     = 0.90
CPU_MODERATE_THRESHOLD = 0.70
CPU_HIGH_PENALTY       = 20
CPU_MODERATE_PENALTY   = 10

MEM_HIGH_THRESHOLD     = 0.90
MEM_MODERATE_THRESHOLD = 0.70
MEM_HIGH_PENALTY       = 20
MEM_MODERATE_PENALTY   = 10

POD_DENSITY_THRESHOLD = 30
POD_DENSITY_PENALTY   = 5

# Headroom weights (60/40 CPU/memory — matches teacher formula)
CPU_HEADROOM_WEIGHT = 0.60
MEM_HEADROOM_WEIGHT = 0.40

# Acceptable-score threshold for candidate preference
MIN_ACCEPTABLE_SCORE = 5   # lowered from 20: piecewise scores can be in 0-15 range under stress

# ---------------------------------------------------------------------------
# Node hardware — allocatable values matching generate_dataset.py profiles
# (cpu_milli * 0.95, mem_bytes * 0.92) to stay within model training distribution.
# Using raw round numbers like 4000/8000/16000m or 8/16/32GB is OOD.
# ---------------------------------------------------------------------------
_GB = 1024 * 1024 * 1024
CPU_CAPACITIES_MILLI = [1_900, 3_800, 7_600, 15_200]   # small/medium/large/xlarge alloc
MEM_CAPACITIES_BYTES = [
    int(3.68  * _GB),   # small:   4GB  * 0.92
    int(7.36  * _GB),   # medium:  8GB  * 0.92
    int(14.72 * _GB),   # large:  16GB  * 0.92
    int(29.44 * _GB),   # xlarge: 32GB  * 0.92
]


# ===========================================================================
# Piecewise-linear helper — mirrors _piecewise_linear_penalty() in generate_dataset.py
# ===========================================================================

def _piecewise_linear_penalty(value: float, points: tuple) -> float:
    """Evaluate a monotone piecewise-linear penalty curve."""
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


# ===========================================================================
# Hardened Node / Pod Generation
# ===========================================================================

def generate_node(name, cluster_size):
    alloc_cpu = random.choice(CPU_CAPACITIES_MILLI)
    alloc_mem = random.choice(MEM_CAPACITIES_BYTES)

    # Beta(3.0, 3.0) centers initial load around 50%–75%
    init_cpu_util = random.betavariate(3.0, 3.0)
    init_mem_util = random.betavariate(3.0, 3.0)

    req_cpu = int(alloc_cpu * init_cpu_util)
    req_mem = int(alloc_mem * init_mem_util)

    pod_count = random.randint(8, 20)

    return {
        "node_name":             name,
        "cluster_size":          cluster_size,
        "allocatable_cpu_milli": alloc_cpu,
        "allocatable_mem_bytes": alloc_mem,
        "requested_cpu_milli":   req_cpu,
        "requested_mem_bytes":   req_mem,
        "pod_count":             pod_count,
    }


def generate_pod(pod_id):
    """Generate large, resource-intensive pods (1 to 8 cores, 2 GB to 16 GB RAM)."""
    return {
        "pod_name":      f"heavy-pod-{pod_id}",
        "req_cpu_milli": random.randint(1_000, 8_000),
        "req_mem_bytes": random.randint(2, 16) * 1024 * 1024 * 1024,
    }


def filter_nodes(nodes, pod):
    return [
        node for node in nodes
        if (pod["req_cpu_milli"] <= node["allocatable_cpu_milli"] and
            pod["req_mem_bytes"] <= node["allocatable_mem_bytes"])
    ]


# ===========================================================================
# Feature derivation
# ===========================================================================

def derive_features(node, pod):
    alloc_cpu = node["allocatable_cpu_milli"]
    alloc_mem = node["allocatable_mem_bytes"]

    req_cpu = node["requested_cpu_milli"]
    req_mem = node["requested_mem_bytes"]

    curr_cpu_pct = req_cpu / alloc_cpu if alloc_cpu > 0 else 0.0
    curr_mem_pct = req_mem / alloc_mem if alloc_mem > 0 else 0.0

    proj_cpu = req_cpu + pod["req_cpu_milli"]
    proj_mem = req_mem + pod["req_mem_bytes"]

    proj_cpu_pct = proj_cpu / alloc_cpu if alloc_cpu > 0 else 0.0
    proj_mem_pct = proj_mem / alloc_mem if alloc_mem > 0 else 0.0

    cpu_headroom = max(0.0, 1.0 - proj_cpu_pct)
    mem_headroom = max(0.0, 1.0 - proj_mem_pct)

    resource_balance  = abs(proj_cpu_pct - proj_mem_pct)
    cpu_request_ratio = pod["req_cpu_milli"] / alloc_cpu if alloc_cpu > 0 else 0.0
    mem_request_ratio = pod["req_mem_bytes"] / alloc_mem if alloc_mem > 0 else 0.0
    # packing_density mirrors generate_dataset.py: projected pod count / 110
    packing_density   = (node["pod_count"] + 1) / 110.0

    return {
        "cluster_size":             float(node["cluster_size"]),
        "node_allocatable_cpu":     float(alloc_cpu),
        "node_allocatable_memory":  float(alloc_mem),
        "current_cpu_usage":        float(req_cpu),
        "current_memory_usage":     float(req_mem),
        "current_cpu_percent":      float(curr_cpu_pct),
        "current_memory_percent":   float(curr_mem_pct),
        "current_pod_count":        float(node["pod_count"]),
        "requested_cpu":            float(pod["req_cpu_milli"]),
        "requested_memory":         float(pod["req_mem_bytes"]),
        "projected_cpu_percent":    float(proj_cpu_pct),
        "projected_memory_percent": float(proj_mem_pct),
        "cpu_headroom":             float(cpu_headroom),
        "memory_headroom":          float(mem_headroom),
        "resource_balance":         float(resource_balance),
        "cpu_request_ratio":        float(cpu_request_ratio),
        "memory_request_ratio":     float(mem_request_ratio),
        "packing_density":          float(packing_density),
    }


# ===========================================================================
# Penalty functions
# ===========================================================================

def calculate_piecewise_penalty(node, pod):
    """
    Piecewise-linear penalty — matches generate_dataset.py teacher formula.
    This is what the model was trained to approximate.
    """
    alloc_cpu = node["allocatable_cpu_milli"]
    alloc_mem = node["allocatable_mem_bytes"]

    proj_cpu_pct = (node["requested_cpu_milli"] + pod["req_cpu_milli"]) / alloc_cpu if alloc_cpu > 0 else 0.0
    proj_mem_pct = (node["requested_mem_bytes"] + pod["req_mem_bytes"]) / alloc_mem if alloc_mem > 0 else 0.0

    cpu_penalty     = _piecewise_linear_penalty(proj_cpu_pct, CPU_PENALTY_POINTS)
    mem_penalty     = _piecewise_linear_penalty(proj_mem_pct, MEM_PENALTY_POINTS)
    density_util    = min(1.0, (node["pod_count"] + 1) / 110.0)
    density_penalty = _piecewise_linear_penalty(density_util, DENSITY_PENALTY_POINTS)

    total = cpu_penalty + mem_penalty + density_penalty
    return total, proj_cpu_pct, proj_mem_pct, density_penalty, cpu_penalty, mem_penalty


def legacy_step_penalty(node, pod):
    """
    Hard step-function penalty (old heuristic baseline).
    Kept for comparison — NOT used for AI score correction.
    """
    alloc_cpu = node["allocatable_cpu_milli"]
    alloc_mem = node["allocatable_mem_bytes"]

    proj_cpu = (node["requested_cpu_milli"] + pod["req_cpu_milli"]) / alloc_cpu if alloc_cpu > 0 else 0.0
    proj_mem = (node["requested_mem_bytes"] + pod["req_mem_bytes"]) / alloc_mem if alloc_mem > 0 else 0.0

    cpu_p = CPU_HIGH_PENALTY if proj_cpu > CPU_HIGH_THRESHOLD else (
        CPU_MODERATE_PENALTY if proj_cpu > CPU_MODERATE_THRESHOLD else 0
    )
    mem_p = MEM_HIGH_PENALTY if proj_mem > MEM_HIGH_THRESHOLD else (
        MEM_MODERATE_PENALTY if proj_mem > MEM_MODERATE_THRESHOLD else 0
    )
    den_p = POD_DENSITY_PENALTY if node["pod_count"] > POD_DENSITY_THRESHOLD else 0

    return cpu_p + mem_p + den_p


# ===========================================================================
# Scoring functions
# ===========================================================================

def ai_score_node(model, node, pod):
    """
    Score a node using the AI model.

    The model predicts teacher_score directly — a value that already
    incorporates the piecewise-linear penalty.  We use the raw model output
    as the final ranking score and keep the penalty breakdown as debug
    information only (NOT subtracted from final).
    """
    features = derive_features(node, pod)
    row = [features[col] for col in FEATURE_COLUMNS]
    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    raw = float(np.clip(np.round(model.predict(df)[0]), MIN_SCORE, MAX_SCORE))
    # Penalty breakdown — kept for debug display, NOT subtracted from raw
    penalty, proj_cpu, proj_mem, dp, cp, mp = calculate_piecewise_penalty(node, pod)

    return raw, {
        "ai_score":        raw,
        "final_score":     raw,    # same as ai_score: no double-penalisation
        "cpu_util":        proj_cpu,
        "mem_util":        proj_mem,
        "cpu_penalty":     cp,
        "mem_penalty":     mp,
        "density_penalty": dp,
        "total_penalty":   penalty,
        "pod_count":       node["pod_count"],
    }


def heuristic_score_node(node, pod):
    """
    Teacher-faithful heuristic: piecewise-linear penalties + 60/40 headroom weighting.
    Matches generate_dataset.py teacher_score_node() and is used as the ground-truth baseline.
    """
    alloc_cpu = node["allocatable_cpu_milli"]
    alloc_mem = node["allocatable_mem_bytes"]

    proj_cpu_pct = (node["requested_cpu_milli"] + pod["req_cpu_milli"]) / alloc_cpu if alloc_cpu > 0 else 0.0
    proj_mem_pct = (node["requested_mem_bytes"] + pod["req_mem_bytes"]) / alloc_mem if alloc_mem > 0 else 0.0

    rem_cpu = max(0.0, 1.0 - proj_cpu_pct)
    rem_mem = max(0.0, 1.0 - proj_mem_pct)

    base = int(round((CPU_HEADROOM_WEIGHT * rem_cpu + MEM_HEADROOM_WEIGHT * rem_mem) * 100))

    cpu_p     = _piecewise_linear_penalty(proj_cpu_pct, CPU_PENALTY_POINTS)
    mem_p     = _piecewise_linear_penalty(proj_mem_pct, MEM_PENALTY_POINTS)
    den_util  = min(1.0, (node["pod_count"] + 1) / 110.0)
    den_p     = _piecewise_linear_penalty(den_util, DENSITY_PENALTY_POINTS)

    score = float(np.clip(base - cpu_p - mem_p - den_p, MIN_SCORE, MAX_SCORE))
    return score


def update_node_state(node, pod):
    node["requested_cpu_milli"] += pod["req_cpu_milli"]
    node["requested_mem_bytes"] += pod["req_mem_bytes"]
    node["pod_count"]           += 1


# ===========================================================================
# Stats accumulator
# ===========================================================================

stats = {
    "clusters_tested":    0,
    "total_placements":   0,
    "unschedulable_pods": 0,
    "exact_matches":      0,
    "ai_rank_sum":        0,
    "score_gap_sum":      0.0,
}


def run_test(model, test_id, pods_per_cluster=12):
    num_nodes = random.randint(5, 10)
    print(f"\n{'='*80}")
    print(f" STRESS TEST #{test_id} -- CONGESTED CLUSTER: {num_nodes} NODES | {pods_per_cluster} HEAVY PODS")
    print(f"{'='*80}")

    stats["clusters_tested"] += 1

    nodes = [generate_node(f"node-{i:02d}", num_nodes) for i in range(1, num_nodes + 1)]
    pods  = [generate_pod(i) for i in range(1, pods_per_cluster + 1)]

    for pod in pods:
        cpu_cores = pod["req_cpu_milli"] / 1000.0
        gb = pod["req_mem_bytes"] / (1024 ** 3)
        print(f"\n--- Scheduling {pod['pod_name']} (Demands: {cpu_cores:.3f} Cores, {gb:.1f}GB RAM) ---")

        eligible = filter_nodes(nodes, pod)
        if not eligible:
            print("  ⚠️  REJECTED / UNSCHEDULABLE: All nodes lack sufficient capacity under heavy load.")
            stats["unschedulable_pods"] += 1
            continue

        stats["total_placements"] += 1

        # Teacher-faithful heuristic baseline (piecewise-linear, matches training)
        for node in eligible:
            node["heuristic_score"] = heuristic_score_node(node, pod)
        h_sorted = sorted(eligible, key=lambda n: n["heuristic_score"], reverse=True)
        h_best       = h_sorted[0]
        h_best_score = h_best["heuristic_score"]

        # AI scoring (with piecewise penalty correction)
        details_map = {}
        for node in eligible:
            final, details = ai_score_node(model, node, pod)
            node["ai_final_score"] = final
            details_map[node["node_name"]] = details

        preferred  = [n for n in eligible if n["ai_final_score"] >= MIN_ACCEPTABLE_SCORE]
        candidates = preferred if preferred else eligible
        ai_sorted  = sorted(candidates, key=lambda n: n["ai_final_score"], reverse=True)
        ai_best    = ai_sorted[0]
        wd         = details_map[ai_best["node_name"]]

        if h_best["node_name"] == ai_best["node_name"]:
            stats["exact_matches"] += 1

        ai_rank = next(
            (i + 1 for i, n in enumerate(h_sorted) if n["node_name"] == ai_best["node_name"]),
            len(eligible),
        )
        stats["ai_rank_sum"] += ai_rank

        ai_heuristic_score = next(
            n["heuristic_score"] for n in eligible if n["node_name"] == ai_best["node_name"]
        )
        score_gap = h_best_score - ai_heuristic_score
        stats["score_gap_sum"] += score_gap

        # Legacy step-function penalty for reference
        legacy_pen = legacy_step_penalty(ai_best, pod)

        print(f"  Heuristic Pick  : {h_best['node_name']} (Score: {h_best_score:.2f})")
        print(
            f"  AI Pick         : {ai_best['node_name']} "
            f"(Score: {wd['final_score']:.2f} | PW Penalty[debug]: {wd['total_penalty']:.2f} | Legacy Penalty[debug]: {legacy_pen})"
        )
        print(f"  AI Heuristic Rank: #{ai_rank} | Score Gap: {score_gap:.2f}")

        update_node_state(ai_best, pod)


def print_summary():
    n = stats["total_placements"]
    print(f"\n{'='*80}")
    print(" HARDENED STRESS-TEST SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Clusters tested              : {stats['clusters_tested']}")
    print(f"Successful placements        : {n}")
    print(f"Rejected / Unschedulable     : {stats['unschedulable_pods']}")
    if n > 0:
        print(f"Exact match (AI = heuristic) : {(stats['exact_matches'] / n) * 100:.1f}%")
        print(f"Avg heuristic rank of AI pick: #{stats['ai_rank_sum'] / n:.2f}")
        print(f"Avg score gap from optimum   : {stats['score_gap_sum'] / n:.2f}")


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return None
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


if __name__ == "__main__":
    model = load_model()
    if model is None:
        raise SystemExit(1)

    print("Successfully loaded model. Running HARDENED stress test suite...")
    for i in range(1, 6):
        run_test(model, i, pods_per_cluster=12)

    print_summary()
