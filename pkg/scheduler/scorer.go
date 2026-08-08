package scheduler

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
)

// ──────────────────────────────────────────────────────────────────────────────
// Tunable constants — all threshold and penalty values live here so that
// nothing is hard-coded inside logic functions.
// ──────────────────────────────────────────────────────────────────────────────
const (
	// defaultAIEndpoint is used when NIYOJAK_AI_ENDPOINT is not set.
	defaultAIEndpoint = "http://niyojak-aiservice:8000/score"

	// aiTimeout is the hard deadline for a single AI scoring call.
	// If the AI service does not respond within this window we apply
	// the built-in heuristic fallback score so the pod is never stuck.
	aiTimeout = 10 * time.Millisecond

	// fallbackScore is returned when the AI service is unreachable or slow.
	// 50 is neutral — the binder will pick the first node that tied at this score
	// which preserves round-robin behaviour identical to the default scheduler.
	fallbackScore = 50

	// minAcceptableScore is the minimum AI score a node must reach before it
	// is considered a viable placement target.  Nodes below this threshold are
	// treated as too stressed for new work and excluded from the candidate set.
	//
	// If ALL eligible nodes score below this threshold (e.g. during a cluster-
	// wide stress event) the scheduler falls back to the least-bad node rather
	// than blocking the pod indefinitely — scheduling always makes progress.
	minAcceptableScore = 20

	// ── CPU utilization thresholds ────────────────────────────────────────────
	// cpuHighThreshold is the projected CPU utilization ratio above which a
	// strong penalty is applied (e.g. 0.90 = 90 %).
	cpuHighThreshold = 0.90
	// cpuModerateThreshold is the ratio above which a moderate penalty starts.
	cpuModerateThreshold = 0.70

	// cpuHighPenalty is the score penalty when projected CPU exceeds cpuHighThreshold.
	cpuHighPenalty = int64(20)
	// cpuModeratePenalty is the score penalty when projected CPU exceeds cpuModerateThreshold.
	cpuModeratePenalty = int64(10)

	// ── Memory utilization thresholds ─────────────────────────────────────────
	// memHighThreshold is the projected memory utilization ratio above which a
	// strong penalty is applied.
	memHighThreshold = 0.90
	// memModerateThreshold is the ratio above which a moderate penalty starts.
	memModerateThreshold = 0.70

	// memHighPenalty is the score penalty when projected memory exceeds memHighThreshold.
	memHighPenalty = int64(20)
	// memModeratePenalty is the score penalty when projected memory exceeds memModerateThreshold.
	memModeratePenalty = int64(10)

	// ── Pod density heuristic ─────────────────────────────────────────────────
	// podDensityThreshold is the number of existing pods on a node above which
	// the density penalty kicks in.
	podDensityThreshold = 30
	// podDensityPenalty is the flat penalty applied when pod count exceeds
	// podDensityThreshold.  It is capped (not multiplied), so a very crowded
	// node is penalised by at most this many points.
	podDensityPenalty = int64(5)

	// ── Score bounds ──────────────────────────────────────────────────────────
	minScore = int64(0)
	maxScore = int64(100)
)

// ScoreRequest is the JSON body sent to the AI service for each (pod, node) pair.
type ScoreRequest struct {
	PodName      string            `json:"pod_name"`
	PodNamespace string            `json:"pod_namespace"`
	NodeName     string            `json:"node_name"`
	PodLabels    map[string]string `json:"pod_labels,omitempty"`

	// Resource requests extracted from the first container.
	// Sent so the AI model can factor in workload intensity.
	CPURequestMillicores int64 `json:"cpu_request_millicores"`
	MemoryRequestBytes   int64 `json:"memory_request_bytes"`
}

// ScoreResponse is what the AI service returns.
type ScoreResponse struct {
	// Score is the placement quality of this node for this pod: 0 (worst) to 100 (best).
	Score int64 `json:"score"`
	// Reason is a human-readable explanation logged for observability.
	Reason string `json:"reason"`
	// Source tells us whether the AI model or a fallback heuristic produced the score.
	Source string `json:"source"`
}

// AIScorer calls the niyojak AI Inference Engine to score candidate nodes.
type AIScorer struct {
	endpoint string
	client   *http.Client
	k8sClient kubernetes.Interface
}

// NewAIScorer creates an AIScorer pointed at the given endpoint.
// If endpoint is empty it falls back to the NIYOJAK_AI_ENDPOINT env var,
// then to the in-cluster default service address.
func NewAIScorer(endpoint string, client kubernetes.Interface) *AIScorer {
	if endpoint == "" {
		endpoint = os.Getenv("NIYOJAK_AI_ENDPOINT")
	}
	if endpoint == "" {
		endpoint = defaultAIEndpoint
	}
	return &AIScorer{
		endpoint:  endpoint,
		client:    &http.Client{Timeout: aiTimeout},
		k8sClient: client,
	}
}

// BestNode scores every candidate node for the given pod and returns
// the name of the highest-scoring node along with its score.
// On any AI service failure the node is scored at fallbackScore so
// scheduling always makes progress.
func (s *AIScorer) BestNode(ctx context.Context, pod *v1.Pod, nodes []v1.Node) (string, int64, error) {
	if len(nodes) == 0 {
		return "", 0, fmt.Errorf("no candidate nodes provided")
	}

	type result struct {
		nodeName string
		score    int64
	}

	// Score every candidate node.
	scores := make([]result, len(nodes))
	for i, node := range nodes {
		sc := s.scoreNode(ctx, pod, &node)
		scores[i] = result{nodeName: node.Name, score: sc}
		klog.V(4).Infof("[niyojak] node %s scored %d/100 for pod %s/%s",
			node.Name, sc, pod.Namespace, pod.Name)
	}

	// Apply minimum score threshold: prefer nodes scoring at or above
	// minAcceptableScore.  This prevents scheduling onto severely stressed nodes
	// when healthier alternatives exist.
	// If every node is below the threshold we fall back to the global least-bad
	// node rather than blocking the pod indefinitely.
	var preferred []result
	for _, r := range scores {
		if r.score >= minAcceptableScore {
			preferred = append(preferred, r)
		}
	}
	candidates := scores
	if len(preferred) > 0 {
		candidates = preferred
		klog.V(3).Infof(
			"[niyojak] pod %s/%s: %d/%d nodes above min-score threshold (%d)",
			pod.Namespace, pod.Name, len(preferred), len(scores), minAcceptableScore,
		)
	} else {
		klog.Warningf(
			"[niyojak] pod %s/%s: all %d nodes score below threshold %d — scheduling on least-bad node",
			pod.Namespace, pod.Name, len(scores), minAcceptableScore,
		)
	}

	// Pick the highest-scoring node from the candidate set.
	// On a tie the first node in the list wins (same as K8s default).
	best := candidates[0]
	for _, r := range candidates[1:] {
		if r.score > best.score {
			best = r
		}
	}

	return best.nodeName, best.score, nil
}

// scoreNode calls the AI service for a single (pod, node) pair,
// then applies a heuristic penalty based on projected utilization.
// It always returns a valid score — on any error it returns fallbackScore.
func (s *AIScorer) scoreNode(ctx context.Context, pod *v1.Pod, node *v1.Node) int64 {
	nodeName := node.Name
	req := s.buildRequest(pod, nodeName)

	body, err := json.Marshal(req)
	if err != nil {
		klog.Warningf("[niyojak] marshal error for node %s: %v — using fallback", nodeName, err)
		return fallbackScore
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.endpoint, bytes.NewReader(body))
	if err != nil {
		klog.Warningf("[niyojak] request creation error for node %s: %v — using fallback", nodeName, err)
		return fallbackScore
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(httpReq)
	if err != nil {
		klog.Warningf("[niyojak] AI service unreachable for node %s: %v — using fallback", nodeName, err)
		return fallbackScore
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		klog.Warningf("[niyojak] AI service returned HTTP %d for node %s — using fallback", resp.StatusCode, nodeName)
		return fallbackScore
	}

	var scoreResp ScoreResponse
	if err := json.NewDecoder(resp.Body).Decode(&scoreResp); err != nil {
		klog.Warningf("[niyojak] decode error for node %s: %v — using fallback", nodeName, err)
		return fallbackScore
	}

	aiScore := scoreResp.Score

	// Calculate and apply heuristic penalty.
	penalty, cpuUtil, memUtil, densityPenalty, cpuPenalty, memPenalty :=
		s.calculateHeuristicPenalty(ctx, pod, node)

	finalScore := aiScore - penalty

	// Clamp defensively — never trust external input unconditionally.
	if finalScore < minScore {
		finalScore = minScore
	}
	if finalScore > maxScore {
		finalScore = maxScore
	}

	// Structured log with all scoring components for observability.
	klog.V(2).Infof(
		"[niyojak] score node=%s pod=%s/%s | ai_score=%d source=%s | cpu_util=%.1f%% mem_util=%.1f%% | penalties: cpu=%d mem=%d density=%d total=%d | final=%d | reason=%q",
		nodeName, pod.Namespace, pod.Name,
		aiScore, scoreResp.Source,
		cpuUtil*100, memUtil*100,
		cpuPenalty, memPenalty, densityPenalty, penalty,
		finalScore,
		scoreResp.Reason,
	)

	return finalScore
}

// heuristicResult groups all intermediate values from calculateHeuristicPenalty
// so that scoreNode can log them without a second computation pass.
// calculateHeuristicPenalty computes penalties for CPU/memory projected utilization
// and pod density to prevent node saturation during pod storms.
//
// Returns:
//
//	totalPenalty  – sum of all penalties (to be subtracted from the AI score)
//	cpuUtil       – projected CPU utilization ratio (0–1)
//	memUtil       – projected memory utilization ratio (0–1)
//	densityPenalty – the pod-density component of the penalty
//	cpuPenalty     – the CPU component of the penalty
//	memPenalty     – the memory component of the penalty
func (s *AIScorer) calculateHeuristicPenalty(ctx context.Context, pod *v1.Pod, node *v1.Node) (
	totalPenalty int64,
	cpuUtil float64,
	memUtil float64,
	densityPenalty int64,
	cpuPenalty int64,
	memPenalty int64,
) {
	// 1. Get running pods on the node via the Kubernetes API.
	fs := fmt.Sprintf("spec.nodeName=%s,status.phase!=Succeeded,status.phase!=Failed", node.Name)
	podList, err := s.k8sClient.CoreV1().Pods("").List(ctx, metav1.ListOptions{FieldSelector: fs})
	if err != nil {
		klog.Warningf("[niyojak] failed to list pods on node %s for heuristics: %v", node.Name, err)
		return 0, 0, 0, 0, 0, 0
	}

	// 2. Sum up resources currently requested on the node.
	var currentCPUMilli int64
	var currentMemBytes int64
	for _, p := range podList.Items {
		for _, c := range p.Spec.Containers {
			if cpu, ok := c.Resources.Requests[v1.ResourceCPU]; ok {
				currentCPUMilli += cpu.MilliValue()
			}
			if mem, ok := c.Resources.Requests[v1.ResourceMemory]; ok {
				currentMemBytes += mem.Value()
			}
		}
	}

	// 3. Get incoming pod's resource requests.
	var podCPUMilli int64
	var podMemBytes int64
	for _, c := range pod.Spec.Containers {
		if cpu, ok := c.Resources.Requests[v1.ResourceCPU]; ok {
			podCPUMilli += cpu.MilliValue()
		}
		if mem, ok := c.Resources.Requests[v1.ResourceMemory]; ok {
			podMemBytes += mem.Value()
		}
	}

	// 4. Calculate projected utilization ratios.
	allocCPU := node.Status.Allocatable[v1.ResourceCPU]
	allocMem := node.Status.Allocatable[v1.ResourceMemory]
	allocCPUMilli := allocCPU.MilliValue()
	allocMemBytes := allocMem.Value()

	if allocCPUMilli > 0 {
		cpuUtil = float64(currentCPUMilli+podCPUMilli) / float64(allocCPUMilli)
	}
	if allocMemBytes > 0 {
		memUtil = float64(currentMemBytes+podMemBytes) / float64(allocMemBytes)
	}

	// 5. Apply tiered CPU penalty using named thresholds.
	//    90 %+ → strong penalty; 70 %+ → moderate penalty.
	switch {
	case cpuUtil > cpuHighThreshold:
		cpuPenalty = cpuHighPenalty
	case cpuUtil > cpuModerateThreshold:
		cpuPenalty = cpuModeratePenalty
	}

	// 6. Apply tiered memory penalty using named thresholds.
	//    90 %+ → strong penalty; 70 %+ → moderate penalty.
	switch {
	case memUtil > memHighThreshold:
		memPenalty = memHighPenalty
	case memUtil > memModerateThreshold:
		memPenalty = memModeratePenalty
	}

	// 7. Apply capped pod-density penalty.
	//    Only applied when the existing pod count exceeds podDensityThreshold.
	//    The penalty is a flat cap (podDensityPenalty) rather than a multiplier,
	//    so an extremely dense node is not penalised disproportionately.
	podCount := len(podList.Items)
	if podCount > podDensityThreshold {
		densityPenalty = podDensityPenalty
	}

	totalPenalty = cpuPenalty + memPenalty + densityPenalty
	return totalPenalty, cpuUtil, memUtil, densityPenalty, cpuPenalty, memPenalty
}

// buildRequest extracts the pod's first container's resource requests and
// constructs the payload sent to the AI service.
func (s *AIScorer) buildRequest(pod *v1.Pod, nodeName string) ScoreRequest {
	req := ScoreRequest{
		PodName:      pod.Name,
		PodNamespace: pod.Namespace,
		NodeName:     nodeName,
		PodLabels:    pod.Labels,
	}

	if len(pod.Spec.Containers) > 0 {
		res := pod.Spec.Containers[0].Resources.Requests
		if cpu, ok := res[v1.ResourceCPU]; ok {
			req.CPURequestMillicores = cpu.MilliValue()
		}
		if mem, ok := res[v1.ResourceMemory]; ok {
			req.MemoryRequestBytes = mem.Value()
		}
	}

	return req
}
