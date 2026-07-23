# `tools/` — Demo Tooling

This directory contains two standalone tools used exclusively during the live demo. They have no dependency on the scheduler or AI engine — they operate purely at the infrastructure and HTTP layer respectively.

These tools are **not part of the core niyojak-scheduler engine**. They exist to create observable real-world conditions that the AI engine then responds to.

---

## `saturate/` — niyojak-saturate

A self-contained Go binary that consumes a configurable percentage of CPU and a fixed amount of RAM on the node it runs on.

**Role in the demo**: Deployed as a Kubernetes Job on a target worker node (via the `/admin` portal). Within seconds, the node CPU spikes to 85%+. The AI feature store observes this through Prometheus telemetry and lowers the node's placement score. The scheduler then routes all new pods away from that node automatically.

**How it is triggered**: The `/admin` portal calls the backend `stress_controller.js` which creates a Kubernetes Job that runs this binary on the chosen node via a `nodeSelector`.

Build:

```bash
GOOS=linux GOARCH=amd64 go build -o bin/niyojak-saturate ./tools/saturate
```

---

## `loadgen/` — niyojak-loadgen

An HTTP load generator that floods concurrent requests against the To-Do App API.

**Role in the demo**: The presenter clicks "Flood Requests" on the `/admin` portal. The loadgen fires hundreds of concurrent HTTP requests per second at the To-Do App. CPU on the existing pods rises above the HPA threshold, causing the **Horizontal Pod Autoscaler (HPA)** to automatically request new pod replicas. Those new pods are then placed by the niyojak-scheduler.

This is intentional — **pods are never created manually**. The HPA is the Kubernetes-native autoscaling primitive that responds to real load. The loadgen simply generates that real load.

**Flow**:
```
loadgen floods HTTP requests
        |
        v
To-Do App pod CPU rises above HPA threshold
        |
        v
HPA requests N new replicas from K8s API
        |
        v
niyojak-scheduler places each new pod on the highest-scoring healthy node
```

Build:

```bash
GOOS=linux GOARCH=amd64 go build -o bin/niyojak-loadgen ./tools/loadgen
```
