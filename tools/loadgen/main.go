// niyojak-loadgen is the HTTP request flood generator for the niyojak demo.
//
// It fires concurrent HTTP requests against the To-Do App API to drive real
// application-level CPU load. When the load is high enough, the Kubernetes
// Horizontal Pod Autoscaler (HPA) automatically requests new pod replicas.
// Those new pods are then placed by niyojak-scheduler using AI node scoring.
//
// The loadgen has NO knowledge of the scheduler or AI service.
// It is purely a traffic generator — its only job is to create real workload
// so the HPA fires organically, rather than manually scaling replicas.
//
// Usage:
//
//	niyojak-loadgen \
//	  --target http://todo-app.default.svc.cluster.local:3000 \
//	  --rps 300 \
//	  --concurrency 50 \
//	  --duration 120s
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// ---------------------------------------------------------------------------
// Flags
// ---------------------------------------------------------------------------

var (
	target      = flag.String("target", "", "Base URL of the To-Do App (required). Example: http://todo-app:3000")
	rps         = flag.Int("rps", 100, "Target requests per second across all workers")
	concurrency = flag.Int("concurrency", 20, "Number of concurrent worker goroutines")
	duration    = flag.Duration("duration", 60*time.Second, "How long to run the flood. 0 means run until SIGTERM.")
	reportEvery = flag.Duration("report", 5*time.Second, "Print a stats line every this interval")
)

// ---------------------------------------------------------------------------
// Shared counters (atomic — safe for concurrent goroutines)
// ---------------------------------------------------------------------------

var (
	totalSent    atomic.Int64
	totalSuccess atomic.Int64
	totalFailed  atomic.Int64
	totalLatency atomic.Int64 // nanoseconds, sum
)

// ---------------------------------------------------------------------------
// Worker pool
// ---------------------------------------------------------------------------

// work is sent from the rate-limiter goroutine to workers via a channel.
type work struct{}

// worker receives tasks and fires HTTP requests until ctx is cancelled.
func worker(ctx context.Context, client *http.Client, baseURL string, jobs <-chan work) {
	for {
		select {
		case <-ctx.Done():
			return
		case _, ok := <-jobs:
			if !ok {
				return
			}
			sendRequest(ctx, client, baseURL)
		}
	}
}

// ---------------------------------------------------------------------------
// Request logic — mixed read/write workload matching real To-Do App usage
// ---------------------------------------------------------------------------

type todoPayload struct {
	Title string `json:"title"`
}

// sendRequest picks a random operation and fires it against the To-Do API.
// Operations are weighted to simulate realistic user behaviour:
//   60% GET  /api/todos          (list all — cheap, high read throughput)
//   30% POST /api/todos          (create a task — generates write + DB work)
//   10% DELETE /api/todos/random (remove a random task)
func sendRequest(ctx context.Context, client *http.Client, baseURL string) {
	start := time.Now()
	totalSent.Add(1)

	var (
		resp *http.Response
		err  error
		roll = rand.Intn(100)
	)

	switch {
	case roll < 60:
		// GET /api/todos
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/api/todos", nil)
		resp, err = client.Do(req)

	case roll < 90:
		// POST /api/todos
		body := todoPayload{Title: fmt.Sprintf("load-test-task-%d", time.Now().UnixNano())}
		buf, _ := json.Marshal(body)
		req, _ := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/api/todos",
			bytes.NewReader(buf))
		req.Header.Set("Content-Type", "application/json")
		resp, err = client.Do(req)

	default:
		// DELETE /api/todos/1 — attempts to delete task ID 1.
		// May 404 if it does not exist — that is expected and counted as success.
		req, _ := http.NewRequestWithContext(ctx, http.MethodDelete,
			baseURL+"/api/todos/1", nil)
		resp, err = client.Do(req)
	}

	elapsed := time.Since(start)
	totalLatency.Add(elapsed.Nanoseconds())

	if err != nil || (resp != nil && resp.StatusCode >= 500) {
		totalFailed.Add(1)
	} else {
		totalSuccess.Add(1)
	}

	if resp != nil && resp.Body != nil {
		resp.Body.Close()
	}
}

// ---------------------------------------------------------------------------
// Stats reporter
// ---------------------------------------------------------------------------

func reporter(ctx context.Context, interval time.Duration) {
	var prevSent int64
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			sent := totalSent.Load()
			success := totalSuccess.Load()
			failed := totalFailed.Load()
			latNs := totalLatency.Load()

			windowSent := sent - prevSent
			prevSent = sent

			var avgMs float64
			if sent > 0 {
				avgMs = float64(latNs) / float64(sent) / 1e6
			}

			actualRPS := float64(windowSent) / interval.Seconds()

			fmt.Printf(
				"[loadgen] sent=%d success=%d failed=%d actual_rps=%.0f avg_latency=%.1fms\n",
				sent, success, failed, actualRPS, avgMs,
			)
		}
	}
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	flag.Parse()

	if *target == "" {
		fmt.Fprintln(os.Stderr, "error: --target is required")
		flag.Usage()
		os.Exit(1)
	}

	// Root context — cancelled on SIGTERM or when duration elapses.
	rootCtx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	var ctx context.Context
	if *duration > 0 {
		var timeoutCancel context.CancelFunc
		ctx, timeoutCancel = context.WithTimeout(rootCtx, *duration)
		defer timeoutCancel()
	} else {
		ctx = rootCtx
	}

	// Shared HTTP client — reuse connections across workers for efficiency.
	client := &http.Client{
		Timeout: 5 * time.Second,
		Transport: &http.Transport{
			MaxIdleConnsPerHost:   *concurrency + 10,
			IdleConnTimeout:       30 * time.Second,
			ResponseHeaderTimeout: 4 * time.Second,
		},
	}

	jobs := make(chan work, *concurrency*2)

	// Launch worker pool.
	var wg sync.WaitGroup
	for i := 0; i < *concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			worker(ctx, client, *target, jobs)
		}()
	}

	// Start stats reporter.
	go reporter(ctx, *reportEvery)

	fmt.Printf(
		"[loadgen] starting: target=%s rps=%d concurrency=%d duration=%s\n",
		*target, *rps, *concurrency, *duration,
	)

	// Rate limiter — feeds the worker pool at the requested RPS.
	// Uses a ticker with interval = 1s / rps to space requests evenly.
	interval := time.Second / time.Duration(*rps)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			close(jobs)
			wg.Wait()
			printFinalStats()
			return
		case <-ticker.C:
			select {
			case jobs <- work{}:
			default:
				// Workers are saturated — drop this tick rather than blocking.
				// This can happen when the target is slow. It is expected behaviour.
			}
		}
	}
}

// printFinalStats prints a summary after the flood ends.
func printFinalStats() {
	sent := totalSent.Load()
	success := totalSuccess.Load()
	failed := totalFailed.Load()
	latNs := totalLatency.Load()

	var avgMs float64
	if sent > 0 {
		avgMs = float64(latNs) / float64(sent) / 1e6
	}

	fmt.Printf(
		"\n[loadgen] done — total_sent=%d success=%d failed=%d avg_latency=%.1fms\n",
		sent, success, failed, avgMs,
	)
}
