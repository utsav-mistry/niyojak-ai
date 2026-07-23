// Package scheduler implements the niyojak-scheduler: a from-scratch Kubernetes
// scheduler that watches pending pods and binds them to nodes scored by the
// niyojak AI Inference Engine.
//
// It does NOT use the kube-scheduler framework. It communicates directly with
// the Kubernetes API server via client-go, making it fully compatible with
// both standard Kubernetes and k3s without any cluster-side patches.
//
// Scheduling loop:
//  1. Watch pods where spec.schedulerName == "niyojak-scheduler" and spec.nodeName == ""
//  2. List all Ready nodes
//  3. Filter nodes: taint/toleration + nodeAffinity + resource capacity check
//  4. If no nodes pass: mark pod Unschedulable (PodCondition + FailedScheduling event) and stop
//  5. POST each eligible (pod, node) to the AI service for a placement score (0-100)
//  6. Bind the pod to the highest-scoring node via the K8s Binding API
//  7. Emit a Kubernetes Event recording which node was chosen and its AI score
package scheduler

import (
	"context"
	"fmt"
	"strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

const (
	// SchedulerName is the value pods must set in spec.schedulerName to opt into niyojak.
	SchedulerName = "niyojak-scheduler"

	// resyncPeriod 0 means the informer only fires on real API events, no full re-list.
	resyncPeriod = 0
)

// Scheduler is the niyojak from-scratch scheduler.
// It owns the watch loop, node filtering, AI scoring, and pod binding.
type Scheduler struct {
	client kubernetes.Interface
	scorer *AIScorer
	binder *Binder
}

// New creates a Scheduler ready to run.
func New(client kubernetes.Interface, aiEndpoint string) *Scheduler {
	return &Scheduler{
		client: client,
		scorer: NewAIScorer(aiEndpoint),
		binder: NewBinder(client),
	}
}

// Run starts the scheduling loop and blocks until ctx is cancelled.
func (s *Scheduler) Run(ctx context.Context) error {
	factory := informers.NewSharedInformerFactory(s.client, resyncPeriod)
	podInformer := factory.Core().V1().Pods().Informer()

	// Only process pods that are unscheduled and addressed to our scheduler.
	podInformer.AddEventHandler(cache.FilteringResourceEventHandler{
		FilterFunc: func(obj interface{}) bool {
			pod, ok := obj.(*v1.Pod)
			if !ok {
				return false
			}
			return pod.Spec.SchedulerName == SchedulerName &&
				pod.Spec.NodeName == "" &&
				pod.Status.Phase != v1.PodSucceeded &&
				pod.Status.Phase != v1.PodFailed
		},
		Handler: cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				pod := obj.(*v1.Pod)
				if err := s.schedule(ctx, pod); err != nil {
					klog.Errorf("[niyojak] scheduling failed for pod %s/%s: %v", pod.Namespace, pod.Name, err)
				}
			},
		},
	})

	factory.Start(ctx.Done())
	if !cache.WaitForCacheSync(ctx.Done(), podInformer.HasSynced) {
		return fmt.Errorf("timed out waiting for pod cache to sync")
	}

	klog.Infof("[niyojak] scheduler running — watching pods with schedulerName=%s", SchedulerName)
	<-ctx.Done()
	return nil
}

// schedule runs the full pipeline for one pending pod:
// list nodes -> filter -> AI score -> bind -> emit event.
func (s *Scheduler) schedule(ctx context.Context, pod *v1.Pod) error {
	klog.V(2).Infof("[niyojak] scheduling pod %s/%s", pod.Namespace, pod.Name)

	if pvcNames := podPVCNames(pod); len(pvcNames) > 0 {
		message := fmt.Sprintf(
			"pod uses PVC-backed volumes (%s). niyojak-scheduler binds pods directly, so StorageClasses with WaitForFirstConsumer can deadlock; prefer emptyDir or an immediate-binding StorageClass",
			strings.Join(pvcNames, ", "),
		)
		klog.Warningf("[niyojak] %s/%s: %s", pod.Namespace, pod.Name, message)
		s.binder.EmitWarningEvent(ctx, pod, message)
	}

	// 1. List all nodes that are Ready and not marked unschedulable.
	nodes, err := s.readyNodes(ctx)
	if err != nil {
		return fmt.Errorf("list nodes: %w", err)
	}
	if len(nodes) == 0 {
		return fmt.Errorf("no schedulable nodes in cluster")
	}

	// 2. Filter: remove nodes the pod cannot run on (taints, nodeAffinity, resource capacity).
	eligible := FilterNodes(pod, nodes)
	if len(eligible) == 0 {
		// No nodes passed the filter. Mark the pod Unschedulable so the
		// Deployment controller and HPA stop spawning replacements.
		// Without this, the pod stays in plain Pending and the controllers
		// keep creating new replicas trying to fill the gap — the pod storm.
		msg := fmt.Sprintf(
			"0/%d nodes are available: node affinity / taint / capacity constraints are not satisfiable",
			len(nodes),
		)
		s.markUnschedulable(ctx, pod, msg)
		return fmt.Errorf("pod %s/%s: no nodes passed filter — marked Unschedulable", pod.Namespace, pod.Name)
	}
	klog.V(3).Infof("[niyojak] pod %s/%s: %d/%d nodes eligible after filter",
		pod.Namespace, pod.Name, len(eligible), len(nodes))

	// 3. Score eligible nodes via the AI service (10ms timeout, fallback to 50 on error).
	bestNode, score, err := s.scorer.BestNode(ctx, pod, eligible)
	if err != nil {
		return fmt.Errorf("AI scoring: %w", err)
	}

	// 4. Bind the pod to the winning node via the K8s Binding API.
	if err := s.binder.Bind(ctx, pod, bestNode); err != nil {
		return fmt.Errorf("bind %s/%s to %s: %w", pod.Namespace, pod.Name, bestNode, err)
	}

	// 5. Emit a K8s Event so kubectl describe pod shows the placement reason.
	s.binder.EmitEvent(ctx, pod, bestNode, score)

	klog.Infof("[niyojak] scheduled %s/%s -> %s (AI score: %d/100)",
		pod.Namespace, pod.Name, bestNode, score)
	return nil
}

// markUnschedulable patches the pod's PodScheduled condition to False/Unschedulable
// and emits a FailedScheduling event — exactly what kube-scheduler does when it
// cannot place a pod. This is critical: without it, pods that will never be
// schedulable stay in plain Pending, and the Deployment controller keeps creating
// replacement pods thinking the previous ones are just slow to start.
func (s *Scheduler) markUnschedulable(ctx context.Context, pod *v1.Pod, reason string) {
	// 1. Patch the pod's condition so kubectl and controllers see it properly.
	now := metav1.Now()
	condition := v1.PodCondition{
		Type:               v1.PodScheduled,
		Status:             v1.ConditionFalse,
		Reason:             "Unschedulable",
		Message:            reason,
		LastTransitionTime: now,
	}

	// Build a minimal JSON patch for the pod status condition.
	patch := fmt.Sprintf(
		`{"status":{"conditions":[{"type":"PodScheduled","status":"False","reason":"Unschedulable","message":%q,"lastTransitionTime":%q}]}}`,
		condition.Message,
		now.UTC().Format("2006-01-02T15:04:05Z"),
	)
	_, err := s.client.CoreV1().Pods(pod.Namespace).Patch(
		ctx, pod.Name, types.MergePatchType, []byte(patch),
		metav1.PatchOptions{}, "status",
	)
	if err != nil {
		klog.Warningf("[niyojak] could not patch Unschedulable condition on pod %s/%s: %v", pod.Namespace, pod.Name, err)
	}

	// 2. Emit a FailedScheduling event so `kubectl describe pod` shows why.
	event := &v1.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s.niyojak-failedscheduling", pod.Name),
			Namespace: pod.Namespace,
		},
		InvolvedObject: v1.ObjectReference{
			APIVersion: "v1",
			Kind:       "Pod",
			Name:       pod.Name,
			Namespace:  pod.Namespace,
			UID:        pod.UID,
		},
		Reason:         "FailedScheduling",
		Message:        reason,
		Source:         v1.EventSource{Component: SchedulerName},
		Type:           v1.EventTypeWarning,
		FirstTimestamp: now,
		LastTimestamp:  now,
	}
	if _, err := s.client.CoreV1().Events(pod.Namespace).Create(ctx, event, metav1.CreateOptions{}); err != nil {
		klog.Warningf("[niyojak] could not emit FailedScheduling event for pod %s/%s: %v", pod.Namespace, pod.Name, err)
	}

	klog.Warningf("[niyojak] pod %s/%s marked Unschedulable: %s", pod.Namespace, pod.Name, reason)
}

// readyNodes lists all nodes that are Ready and not explicitly unschedulable.
func (s *Scheduler) readyNodes(ctx context.Context) ([]v1.Node, error) {
	nodeList, err := s.client.CoreV1().Nodes().List(ctx, metav1.ListOptions{
		// spec.unschedulable is not an indexed field in all clusters,
		// so we filter in-process below rather than relying on field selector.
	})
	if err != nil {
		return nil, err
	}

	var ready []v1.Node
	for _, node := range nodeList.Items {
		if !node.Spec.Unschedulable && isNodeReady(&node) {
			ready = append(ready, node)
		}
	}
	return ready, nil
}

// isNodeReady returns true when the node's Ready condition is True.
func isNodeReady(node *v1.Node) bool {
	for _, cond := range node.Status.Conditions {
		if cond.Type == v1.NodeReady {
			return cond.Status == v1.ConditionTrue
		}
	}
	return false
}

// podPVCNames returns the names of PVC-backed volumes used by pod.
func podPVCNames(pod *v1.Pod) []string {
	var pvcNames []string
	for _, volume := range pod.Spec.Volumes {
		if volume.PersistentVolumeClaim != nil {
			claimName := volume.PersistentVolumeClaim.ClaimName
			if claimName == "" {
				claimName = volume.Name
			}
			pvcNames = append(pvcNames, claimName)
		}
	}
	return pvcNames
}
