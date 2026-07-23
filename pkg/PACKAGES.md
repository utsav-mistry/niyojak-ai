# `pkg/` — Shared Go Packages

This directory contains internal Go packages that are imported by the binaries in `cmd/`.

| Package | Import Path | Purpose |
|---|---|---|
| `scheduler/` | `github.com/niyojak/niyojak/pkg/scheduler` | Core K8s scheduling framework plugin. Implements the `Score` extension point that calls the AI Inference Engine. |

## Design Rule

Packages in `pkg/` must **not** contain `main` functions. They are importable libraries that binaries in `cmd/` compose together.

## Testing

```bash
# Run all unit tests for all packages
go test ./pkg/...

# Run with verbose output and race detector
go test -v -race ./pkg/...
```
