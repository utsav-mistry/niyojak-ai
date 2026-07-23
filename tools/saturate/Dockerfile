# syntax=docker/dockerfile:1

FROM golang:1.21-alpine AS builder

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o /bin/niyojak-saturate ./tools/saturate/


FROM scratch

COPY --from=builder /bin/niyojak-saturate /niyojak-saturate

ENTRYPOINT ["/niyojak-saturate"]
