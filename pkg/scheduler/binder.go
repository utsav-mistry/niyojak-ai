package scheduler

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
)

// Binder binds a pod to a node using the Kubernetes Binding subresource API.
// This is exactly what the default kube-scheduler does to place a pod —
// we call the same API endpoint, just driven by our AI scores.
type Binder struct {
	client kubernetes.Interface
}

// NewBinder creates a Binder backed by the given Kubernetes client.
func NewBinder(client kubernetes.Interface) *Binder {
	return &Binder{client: client}
}

// Bind places pod onto nodeName by creating a Binding object against the
// Kubernetes API server. On success the API server sets pod.spec.nodeName
// and the kubelet on that node picks up the pod immediately.
func (b *Binder) Bind(ctx context.Context, pod *v1.Pod, nodeName string) error {
	binding := &v1.Binding{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: pod.Namespace,
			Name:      pod.Name,
			UID:       pod.UID,
		},
		Target: v1.ObjectReference{
			APIVersion: "v1",
			Kind:       "Node",
			Name:       nodeName,
		},
	}

	err := b.client.CoreV1().Pods(pod.Namespace).Bind(ctx, binding, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("binding call failed: %w", err)
	}

	klog.V(2).Infof("[niyojak] bound pod %s/%s to node %s", pod.Namespace, pod.Name, nodeName)
	return nil
}

// EmitEvent writes a Kubernetes Event on the pod recording the scheduling decision.
// This makes the placement visible in `kubectl describe pod` and `kubectl get events`.
func (b *Binder) EmitEvent(ctx context.Context, pod *v1.Pod, nodeName string, score int64) {
	event := &v1.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s.niyojak-scheduled", pod.Name),
			Namespace: pod.Namespace,
		},
		InvolvedObject: v1.ObjectReference{
			APIVersion: "v1",
			Kind:       "Pod",
			Name:       pod.Name,
			Namespace:  pod.Namespace,
			UID:        pod.UID,
		},
		Reason:  "Scheduled",
		Message: fmt.Sprintf("niyojak-scheduler placed pod on node %s (AI score: %d/100)", nodeName, score),
		Source: v1.EventSource{
			Component: SchedulerName,
		},
		Type:           v1.EventTypeNormal,
		FirstTimestamp: metav1.Now(),
		LastTimestamp:  metav1.Now(),
	}

	if _, err := b.client.CoreV1().Events(pod.Namespace).Create(ctx, event, metav1.CreateOptions{}); err != nil {
		// Event failure is non-fatal — the pod is already bound.
		klog.Warningf("[niyojak] could not emit scheduling event for pod %s/%s: %v", pod.Namespace, pod.Name, err)
	}
}

// EmitWarningEvent writes a Kubernetes Warning Event on the pod.
// This is used for non-fatal scheduling problems that need operator visibility.
func (b *Binder) EmitWarningEvent(ctx context.Context, pod *v1.Pod, message string) {
	event := &v1.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s.niyojak-warning", pod.Name),
			Namespace: pod.Namespace,
		},
		InvolvedObject: v1.ObjectReference{
			APIVersion: "v1",
			Kind:       "Pod",
			Name:       pod.Name,
			Namespace:  pod.Namespace,
			UID:        pod.UID,
		},
		Reason:         "PVCBindingRisk",
		Message:        message,
		Source:         v1.EventSource{Component: SchedulerName},
		Type:           v1.EventTypeWarning,
		FirstTimestamp: metav1.Now(),
		LastTimestamp:  metav1.Now(),
	}

	if _, err := b.client.CoreV1().Events(pod.Namespace).Create(ctx, event, metav1.CreateOptions{}); err != nil {
		klog.Warningf("[niyojak] could not emit PVC warning event for pod %s/%s: %v", pod.Namespace, pod.Name, err)
	}
}
