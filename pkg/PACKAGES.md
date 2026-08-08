# `pkg/` — Shared Go Packages

This directory contains internal Go packages imported by the `cmd/scheduler/` binary. All scheduling logic lives here.

## Packages

| Package | Import Path | Purpose |
|---|---|---|
| `scheduler/` | `github.com/utsav-mistry/niyojak/pkg/scheduler` | Complete from-scratch Kubernetes scheduler: watch loop, node filtering, AI scoring, and pod binding |

## Package Contents (`pkg/scheduler/`)

| File | Purpose |
|---|---|
| `scheduler.go` | Core watch loop — uses `client-go` SharedInformer to watch pending pods with `schedulerName: niyojak-scheduler`, then coordinates filter → score → bind |
| `filter.go` | `FilterNodes()` — pre-AI eligibility checks: taint/toleration, `requiredDuringScheduling` node affinity, and resource capacity (`hasCapacity`) |
| `scorer.go` | `AIScorer` + heuristic fallback — HTTP client calling `POST /score` on the AI service with a hard 10ms timeout; applies `minAcceptableScore=20` threshold; falls back to score=50 on timeout |
| `binder.go` | `Binder` — binds the chosen pod to the winning node via the K8s Binding API and emits a Kubernetes Event recording the node and AI score |

## Design Rule

Packages in `pkg/` must **not** contain `main` functions. They are importable libraries that `cmd/scheduler/` composes into the binary.

## Scheduler Architecture (Not a kube-scheduler Plugin)

NIYOJAK is a **from-scratch out-of-tree scheduler** — it does **not** use the `kube-scheduler` plugin framework. It communicates directly with the Kubernetes API server via `client-go` only. This makes it compatible with both k3s and standard Kubernetes (v1.26+) with no cluster-side patches.

Scheduling loop per pod:
1. Watch for pods where `spec.schedulerName == "niyojak-scheduler"` and `spec.nodeName == ""`
2. List all Ready, schedulable nodes
3. Filter: taint/toleration + node affinity + resource capacity
4. If zero nodes pass: mark pod `Unschedulable` (condition + `FailedScheduling` event) and skip
5. POST each eligible `(pod, node)` to AI service for a placement score (0–100)
6. Bind pod to highest-scoring node via K8s Binding API
7. Emit Kubernetes Event: `"Scheduled pod to node X (AI score: 92, source: xgboost)"`

## Testing

```bash
# Run all unit tests
go test ./pkg/...

# With verbose output and race detector
go test -v -race ./pkg/...
```
