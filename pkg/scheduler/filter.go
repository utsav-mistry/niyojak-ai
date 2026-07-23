package scheduler

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

// filter.go — pre-scoring node eligibility checks.
//
// Before calling the AI service we eliminate nodes that cannot physically
// host the pod. This mirrors the Filter stage of the kube-scheduler:
// we check taints/tolerations and basic resource fit.
//
// Nodes that pass all checks here are sent to the AI scorer.
// Nodes that fail are excluded silently — the AI never sees them.

// FilterNodes returns the subset of nodes that are eligible to run pod.
// A node is eligible when:
//   1. It is Ready and not marked unschedulable (already guaranteed by readyNodes() in scheduler.go)
//   2. The pod tolerates all taints the node carries
//   3. The node has enough allocatable CPU and memory for the pod's requests
func FilterNodes(pod *v1.Pod, nodes []v1.Node) []v1.Node {
	var eligible []v1.Node
	for _, node := range nodes {
		if !toleratesTaints(pod, &node) {
			continue
		}
		if !hasCapacity(pod, &node) {
			continue
		}
		eligible = append(eligible, node)
	}
	return eligible
}

// toleratesTaints returns true if the pod's tolerations cover every taint
// on the node. Any NoSchedule or NoExecute taint that is not tolerated
// disqualifies the node.
func toleratesTaints(pod *v1.Pod, node *v1.Node) bool {
	for _, taint := range node.Spec.Taints {
		// Only hard scheduling constraints matter here.
		if taint.Effect == v1.TaintEffectPreferNoSchedule {
			continue
		}
		if !podToleratesToTaint(pod, taint) {
			return false
		}
	}
	return true
}

// podToleratesToTaint checks whether pod has a toleration that matches taint.
func podToleratesToTaint(pod *v1.Pod, taint v1.Taint) bool {
	for _, toleration := range pod.Spec.Tolerations {
		if tolerationMatchesTaint(toleration, taint) {
			return true
		}
	}
	return false
}

// tolerationMatchesTaint returns true if toleration covers taint.
// A toleration with an empty key matches any taint key (wildcard).
// A toleration with operator Exists matches any taint value for that key.
func tolerationMatchesTaint(t v1.Toleration, taint v1.Taint) bool {
	// Effect must match unless the toleration has no effect specified (matches all).
	if t.Effect != "" && t.Effect != taint.Effect {
		return false
	}
	// Key must match unless the toleration key is empty (wildcard).
	if t.Key != "" && t.Key != taint.Key {
		return false
	}
	// Value match depends on operator.
	switch t.Operator {
	case v1.TolerationOpExists:
		// Exists means any value is tolerated.
		return true
	case v1.TolerationOpEqual, "":
		return t.Value == taint.Value
	}
	return false
}

// hasCapacity returns true if node has enough allocatable CPU and memory
// to satisfy the sum of requests across all containers in pod.
func hasCapacity(pod *v1.Pod, node *v1.Node) bool {
	podCPU, podMem := podRequests(pod)

	allocCPU := node.Status.Allocatable[v1.ResourceCPU]
	allocMem := node.Status.Allocatable[v1.ResourceMemory]

	// If allocatable is not set the node has not registered its capacity yet —
	// be conservative and exclude it.
	if allocCPU.IsZero() || allocMem.IsZero() {
		return false
	}

	return podCPU.Cmp(allocCPU) <= 0 && podMem.Cmp(allocMem) <= 0
}

// podRequests returns the sum of CPU and memory requests across all init
// containers and regular containers in pod.
func podRequests(pod *v1.Pod) (cpu, mem resource.Quantity) {
	for _, c := range pod.Spec.InitContainers {
		if req, ok := c.Resources.Requests[v1.ResourceCPU]; ok {
			cpu.Add(req)
		}
		if req, ok := c.Resources.Requests[v1.ResourceMemory]; ok {
			mem.Add(req)
		}
	}
	for _, c := range pod.Spec.Containers {
		if req, ok := c.Resources.Requests[v1.ResourceCPU]; ok {
			cpu.Add(req)
		}
		if req, ok := c.Resources.Requests[v1.ResourceMemory]; ok {
			mem.Add(req)
		}
	}
	return cpu, mem
}
