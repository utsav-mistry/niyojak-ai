# `cmd/` — Scheduler Binary Entrypoint

This directory contains only the `main` package for the **niyojak-scheduler** binary. All scheduling logic is in `pkg/scheduler/`.

The demo tools (saturate, loadgen) are completely separate and live in `tools/`.

## `scheduler/`

| File | Purpose |
|---|---|
| `main.go` | Binary entrypoint — parses `--ai-endpoint` and `--v` flags, builds the `client-go` Kubernetes client, instantiates `pkg/scheduler.Scheduler`, and calls `Run()` |
| `Dockerfile` | Minimal Alpine container image for cluster deployment |

Compiles into the `niyojak-scheduler` binary — a from-scratch out-of-tree Kubernetes scheduler that:
- Watches pending pods with `schedulerName: niyojak-scheduler`
- Runs filter (taint/toleration + node affinity + resource capacity) before AI scoring
- Calls `POST /score` on the AI service for each eligible node
- Binds the pod to the highest-scoring node
- Emits a Kubernetes Event on every successful bind
- Falls back to neutral score (50) if the AI service is unreachable, ensuring pod scheduling **never blocks**

## Build

```bash
# Linux target (for deployment inside cluster)
GOOS=linux GOARCH=amd64 go build -o bin/niyojak-scheduler ./cmd/scheduler

# Local build (for testing)
go build -o bin/niyojak-scheduler ./cmd/scheduler
```

## Runtime Flags

| Flag | Default | Description |
|---|---|---|
| `--ai-endpoint` | `http://niyojak-aiservice:8000/score` | AI service scoring URL |
| `--v` | `2` | klog verbosity level |

## Notes

- Compatible with both k3s and standard Kubernetes (v1.26+)
- Works alongside the default scheduler — only pods that opt in via `schedulerName: niyojak-scheduler` are intercepted
- No dependency on `tools/` — the scheduler has zero knowledge of the saturate or loadgen tools
- Pinned to the control-plane node via `nodeSelector` in the deployment manifest (so it survives worker stress events)
