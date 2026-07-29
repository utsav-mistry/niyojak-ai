package scheduler

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

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
}

// NewAIScorer creates an AIScorer pointed at the given endpoint.
// If endpoint is empty it falls back to the NIYOJAK_AI_ENDPOINT env var,
// then to the in-cluster default service address.
func NewAIScorer(endpoint string) *AIScorer {
	if endpoint == "" {
		endpoint = os.Getenv("NIYOJAK_AI_ENDPOINT")
	}
	if endpoint == "" {
		endpoint = defaultAIEndpoint
	}
	return &AIScorer{
		endpoint: endpoint,
		client:   &http.Client{Timeout: aiTimeout},
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
		sc := s.scoreNode(ctx, pod, node.Name)
		scores[i] = result{nodeName: node.Name, score: sc}
		klog.V(4).Infof("[niyojak] node %s scored %d/100 for pod %s/%s",
			node.Name, sc, pod.Namespace, pod.Name)
	}

	// Apply minimum score threshold: prefer nodes scoring at or above
	// minAcceptableScore.  This prevents scheduling onto severely stressed nodes
	// when healthier alternatives exist (Gap 4 fix).
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

// scoreNode calls the AI service for a single (pod, node) pair.
// It always returns a valid score — on any error it returns fallbackScore.
func (s *AIScorer) scoreNode(ctx context.Context, pod *v1.Pod, nodeName string) int64 {
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

	score := scoreResp.Score
	// Clamp defensively — never trust external input unconditionally.
	if score < 0 {
		score = 0
	}
	if score > 100 {
		score = 100
	}

	klog.V(2).Infof("[niyojak] AI scored node %s: %d (%s) — %s", nodeName, score, scoreResp.Source, scoreResp.Reason)
	return score
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
