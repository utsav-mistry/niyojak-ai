// niyojak-scheduler is an AI-powered Kubernetes scheduler built from scratch.
//
// It watches the Kubernetes API for pending pods that carry
//   schedulerName: niyojak-scheduler
// filters candidate nodes for taint/toleration and resource capacity,
// calls the niyojak AI Inference Engine to score each node,
// and binds the pod to the highest-scoring node via the K8s Binding API.
//
// Works on both k3s and standard Kubernetes without any cluster-side patches.
// The only requirement is a ServiceAccount with the RBAC permissions listed
// in the Helm chart (charts/niyojak/templates/rbac.yaml).
//
// Usage:
//
//	niyojak-scheduler \
//	  --kubeconfig /path/to/kubeconfig \   # omit when running in-cluster
//	  --ai-endpoint http://niyojak-aiservice:8000/score
package main

import (
	"context"
	"flag"
	"os"
	"os/signal"
	"syscall"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog/v2"

	"github.com/niyojak/niyojak/pkg/scheduler"
)

func main() {
	// --- Flags -----------------------------------------------------------
	var (
		kubeconfig  string
		aiEndpoint  string
	)

	klog.InitFlags(nil)
	flag.StringVar(&kubeconfig, "kubeconfig", "",
		"Path to kubeconfig file. Leave empty to use in-cluster config (default when running inside a pod).")
	flag.StringVar(&aiEndpoint, "ai-endpoint", "",
		"URL of the niyojak AI Inference Engine /score endpoint. "+
			"Falls back to NIYOJAK_AI_ENDPOINT env var, then http://niyojak-aiservice:8000/score.")
	flag.Parse()

	// --- Kubernetes client -----------------------------------------------
	cfg, err := buildConfig(kubeconfig)
	if err != nil {
		klog.Fatalf("build kubeconfig: %v", err)
	}

	client, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		klog.Fatalf("create kubernetes client: %v", err)
	}

	// --- Scheduler -------------------------------------------------------
	sched := scheduler.New(client, aiEndpoint)

	// --- Graceful shutdown via SIGTERM / SIGINT --------------------------
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, os.Interrupt)
	defer stop()

	klog.Info("niyojak-scheduler starting")
	if err := sched.Run(ctx); err != nil {
		klog.Fatalf("scheduler exited with error: %v", err)
	}
	klog.Info("niyojak-scheduler stopped cleanly")
}

// buildConfig returns a *rest.Config from kubeconfig file (dev/local use)
// or from the in-cluster environment (production, running as a pod).
func buildConfig(kubeconfig string) (*rest.Config, error) {
	if kubeconfig != "" {
		return clientcmd.BuildConfigFromFlags("", kubeconfig)
	}

	// Running inside the cluster — use the service account token mounted
	// at /var/run/secrets/kubernetes.io/serviceaccount/token.
	cfg, err := rest.InClusterConfig()
	if err == nil {
		return cfg, nil
	}

	// Fallback: try KUBECONFIG env var (useful for local dev without --kubeconfig flag).
	if kc := os.Getenv("KUBECONFIG"); kc != "" {
		return clientcmd.BuildConfigFromFlags("", kc)
	}

	// Last resort: try default kubeconfig path.
	return clientcmd.BuildConfigFromFlags("", clientcmd.RecommendedHomeFile)
}
