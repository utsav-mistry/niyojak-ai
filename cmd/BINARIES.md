# `cmd/` — Scheduler Binary Entrypoint

This directory contains only the `main` package for the **niyojak-scheduler** binary.

The stressor and load generator are demo tooling and live in `tools/` — completely separate from the core scheduler concern.

## `scheduler/`

Compiles into the `niyojak-scheduler` binary — an out-of-tree Kubernetes scheduler plugin that:
- Intercepts pod scheduling for any pod with `schedulerName: niyojak-scheduler`
- Calls the AI Inference Engine (`ai_service/`) to get a node score (0-100) per candidate node
- Returns the highest-scoring node to the K8s API server
- Falls back to a neutral score (50) if the AI service is unreachable, ensuring pod scheduling is never blocked

## Build

```bash
# Linux target (for deployment inside cluster)
GOOS=linux GOARCH=amd64 go build -o bin/niyojak-scheduler ./cmd/scheduler

# Local build (for testing)
go build -o bin/niyojak-scheduler ./cmd/scheduler
```

## Notes

- Compatible with both k3s and standard Kubernetes (v1.26+)
- Works alongside the default scheduler — only pods that opt in via `schedulerName` are intercepted
- No dependency on `tools/` — the scheduler has zero knowledge of the saturate or loadgen tools
