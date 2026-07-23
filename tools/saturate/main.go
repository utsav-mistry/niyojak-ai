package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"os/signal"
	"runtime"
	"sync"
	"syscall"
	"time"
)

// stressor is a self-contained CPU & memory stress tool.
// It is deployed as a DaemonSet-style job on a target node via the /admin portal.
// Usage: stressor --cpu 80 --mem 512 --duration 60
//
//	--cpu: target CPU% to consume (0-100)
//	--mem: megabytes of RAM to allocate and hold
//	--duration: seconds to run (0 = run until SIGTERM)

func main() {
	cpuTarget := flag.Float64("cpu", 80, "target CPU% to consume (0-100)")
	memMB := flag.Int("mem", 256, "megabytes of RAM to allocate and hold")
	duration := flag.Int("duration", 0, "seconds to run (0 = until SIGTERM)")
	flag.Parse()

	fmt.Printf("[stressor] starting: cpu=%.0f%% mem=%dMB duration=%ds\n", *cpuTarget, *memMB, *duration)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	if *duration > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(*duration)*time.Second)
		defer cancel()
	}

	var wg sync.WaitGroup

	// -- Memory stress: allocate a big slice and touch it to prevent GC
	wg.Add(1)
	go func() {
		defer wg.Done()
		size := *memMB * 1024 * 1024
		buf := make([]byte, size)
		// Touch every page so it's actually resident
		for i := 0; i < size; i += 4096 {
			buf[i] = byte(i)
		}
		fmt.Printf("[stressor] holding %dMB of RAM\n", *memMB)
		<-ctx.Done()
		runtime.KeepAlive(buf)
	}()

	// -- CPU stress: spin goroutines doing busy math to hit the target %
	numCPU := runtime.NumCPU()
	wg.Add(numCPU)
	for i := 0; i < numCPU; i++ {
		go func() {
			defer wg.Done()
			burnCPU(ctx, *cpuTarget/100.0)
		}()
	}

	<-ctx.Done()
	fmt.Println("[stressor] shutting down cleanly")
	wg.Wait()
	os.Exit(0)
}

// burnCPU keeps a goroutine busy for `load` fraction of time.
// e.g. load=0.8 → busy 80% of each 10ms window.
func burnCPU(ctx context.Context, load float64) {
	windowNs := 10 * time.Millisecond
	busyNs := time.Duration(float64(windowNs) * load)

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		start := time.Now()
		// Busy loop for `busyNs`
		for time.Since(start) < busyNs {
			_ = math.Sqrt(float64(time.Now().UnixNano()))
		}
		// Sleep for the remainder of the window
		remaining := windowNs - time.Since(start)
		if remaining > 0 {
			time.Sleep(remaining)
		}
	}
}
