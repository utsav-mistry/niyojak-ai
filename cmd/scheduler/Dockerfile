# syntax=docker/dockerfile:1

# Build the niyojak-scheduler binary
FROM golang:1.21-alpine AS builder

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o /bin/niyojak-scheduler ./cmd/scheduler/


# Minimal runtime image — no shell, no package manager
FROM scratch

COPY --from=builder /bin/niyojak-scheduler /niyojak-scheduler

# The scheduler reads KUBECONFIG or uses in-cluster service account automatically.
ENTRYPOINT ["/niyojak-scheduler"]
