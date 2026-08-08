   # Step-by-Step NIYOJAK Setup Guide

   This guide is written for the exact lab layout you described:

   - Ubuntu Desktop host = Node 1 (control plane + worker)
   - Ubuntu Server VM 1 = Node 2 (worker)
   - Ubuntu Server VM 2 = Node 3 (worker)

   The goal is to get the full NIYOJAK demo running end-to-end:

   1. install Kubernetes on the host,
   2. join the two VMs as worker nodes,
   3. deploy the scheduler, AI service, and demo app,
   4. train the AI model,
   5. and run the stress-and-schedule demo.

   ---

   ## 1. Prepare the machine layout

   ### Machine 1 — Ubuntu Desktop host (Node 1)
   Use this machine as the main control plane.

   Recommended minimum:
   - 4 CPU cores
   - 8 GB RAM
   - 50 GB free disk
   - Internet access

   ### Machine 2 and 3 — Ubuntu Server VMs (Node 2 and Node 3)
   Use two Ubuntu Server VMs with:
   - 2 CPU cores each
   - 2 GB RAM each
   - 20 GB disk each

   If your machine is not very powerful, reduce the VM sizes and keep the stress demo light.

   ---

   ## 2. Prepare the Ubuntu Desktop host

   Run these commands on the Ubuntu Desktop machine.

   ### 2.1 Update the system

   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

   ### 2.2 Install base tools

   ```bash
   sudo apt install -y curl jq git snapd ca-certificates gnupg lsb-release
   ```

   ### 2.3 Install Docker (optional but useful for image builds)

   If you want to build container images locally:

   ```bash
   sudo apt install -y docker.io
   sudo systemctl enable docker
   sudo systemctl start docker
   sudo usermod -aG docker $USER
   newgrp docker
   ```

   > If you do not want to build images locally, you can skip this step and use the container images already referenced in the manifests.

   ---

   ## 3. Install Kubernetes on the Ubuntu Desktop host

   ### 3.1 Install k3s as the control plane

   ```bash
   curl -sfL https://get.k3s.io | sh -s - --disable=traefik --write-kubeconfig-mode=644
   ```

   This installs k3s and makes the host a control plane + worker node.

   ### 3.2 Check that k3s is running

   ```bash
   sudo systemctl status k3s --no-pager
   ```

   ### 3.3 Configure kubectl for your user

   ```bash
   mkdir -p ~/.kube
   sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
   sudo chown $USER:$USER ~/.kube/config
   export KUBECONFIG=~/.kube/config
   ```

   ### 3.4 Verify the cluster

   ```bash
   kubectl get nodes
   kubectl get pods -A
   ```

   You should see the Ubuntu Desktop host as a Ready node.

   ---

   ## 4. Install Helm and metrics-server

   ### 4.1 Install Helm

   ```bash
   curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
   ```

   ### 4.2 Install metrics-server for HPA support

   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
   kubectl patch deployment metrics-server -n kube-system --type=json -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]' 2>/dev/null || true
   ```

   ### 4.3 Verify metrics-server

   ```bash
   kubectl get deployment metrics-server -n kube-system
   ```

   ---

   ## 5. Prepare the NIYOJAK repository on the host

   ### 5.1 Clone the repository

   ```bash
   git clone https://github.com/<your-user>/niyojak-ai.git
   cd niyojak-ai
   ```

   If you already have the repository on your Windows machine, copy it into the Ubuntu host using a tool such as SCP or Git clone directly on Ubuntu.

   ### 5.2 Build the Go scheduler binary

   ```bash
   go build ./cmd/scheduler/
   ```

   ### 5.3 Build the stress tool and load generator

   ```bash
   go build ./tools/saturate/
   go build ./tools/loadgen/
   ```

   ---

   ## 6. Build and publish the container images

   The deployment manifests expect container images such as:

   - utsavmistry/niyojak-scheduler:latest
   - utsavmistry/niyojak-aiservice:latest
   - utsavmistry/niyojak-todo-app:latest

   If those images are not available in your environment, build and publish your own images.

   ### 6.1 Build the images

   ```bash
   docker build -t niyojak-scheduler:local -f cmd/scheduler/Dockerfile .
   docker build -t niyojak-aiservice:local -f ai_service/Dockerfile .
   docker build -t niyojak-todo-app:local -f sample_app/Dockerfile .
   ```

   ### 6.2 Push to a registry

   Use any registry reachable from the cluster, such as GHCR.

   Example idea:

   ```bash
   docker tag niyojak-scheduler:local ghcr.io/<your-user>/niyojak-scheduler:latest
   docker tag niyojak-aiservice:local ghcr.io/<your-user>/niyojak-aiservice:latest
   docker tag niyojak-todo-app:local ghcr.io/<your-user>/niyojak-todo-app:latest

   docker push ghcr.io/<your-user>/niyojak-scheduler:latest
   docker push ghcr.io/<your-user>/niyojak-aiservice:latest
   docker push ghcr.io/<your-user>/niyojak-todo-app:latest
   ```

   Then update the image names in the manifests if needed.

   > If you want the simplest path, use the default image names from the repo first and only rebuild if pulls fail.

   ---

   ## 7. Create the two worker VMs

   You can do this from the Ubuntu Desktop host using Multipass.

   ### 7.1 Install Multipass

   ```bash
   sudo snap install multipass
   ```

   ### 7.2 Launch VM 1 and VM 2

   ```bash
   multipass launch 22.04 --name niyojak-node2 --cpus 2 --memory 2G --disk 10G
   multipass launch 22.04 --name niyojak-node3 --cpus 2 --memory 2G --disk 10G
   ```

   ### 7.3 Check the VMs

   ```bash
   multipass list
   ```

   ---

   ## 8. Join the VMs to the k3s cluster

   ### 8.1 Get the join token from the host

   ```bash
   sudo cat /var/lib/rancher/k3s/server/node-token
   ```

   Save the token somewhere safe.

   ### 8.2 Get the host IP address

   ```bash
   hostname -I
   ```

   The first IP shown is usually the host address to use for joining.

   ### 8.3 Run the join command inside each VM

   Example:

   ```bash
   multipass exec niyojak-node2 -- bash -c 'curl -sfL https://get.k3s.io | K3S_URL="https://<HOST_IP>:6443" K3S_TOKEN="<JOIN_TOKEN>" sh -s - agent'
   ```

   ```bash
   multipass exec niyojak-node3 -- bash -c 'curl -sfL https://get.k3s.io | K3S_URL="https://<HOST_IP>:6443" K3S_TOKEN="<JOIN_TOKEN>" sh -s - agent'
   ```

   Replace `<HOST_IP>` and `<JOIN_TOKEN>` with the actual values.

   ### 8.4 Verify the nodes appear in the cluster

   Back on the Ubuntu Desktop host:

   ```bash
   kubectl get nodes -o wide
   ```

   You should see:
   - one control-plane node (the host)
   - one worker node from VM 1
   - one worker node from VM 2

   ---

   ## 9. Deploy the observability stack

   ### 9.1 Apply node-exporter, Prometheus, and Grafana

   ```bash
   kubectl apply -f deploy/observability/node-exporter.yaml
   kubectl apply -f deploy/observability/prometheus.yaml
   kubectl apply -f deploy/observability/grafana.yaml
   ```

   ### 9.2 Verify the pods

   ```bash
   kubectl get pods -n niyojak-system
   ```

   You should see the monitoring stack pods coming up.

   ---

   ## 10. Deploy the NIYOJAK scheduler and AI service

   ### 10.1 Apply RBAC

   ```bash
   kubectl apply -f deploy/manifests/rbac.yaml
   ```

   ### 10.2 Apply the system manifests

   ```bash
   kubectl apply -f deploy/manifests/niyojak-system.yaml
   ```

   ### 10.3 Check the system pods

   ```bash
   kubectl get pods -n niyojak-system
   kubectl logs -n niyojak-system deploy/niyojak-aiservice --tail=100
   kubectl logs -n niyojak-system deploy/niyojak-scheduler --tail=100
   ```

   ---

   ## 11. Deploy the demo app

   ### 11.1 Apply the To-Do app deployment

   ```bash
   kubectl apply -f sample_app/todo-app-deployment.yaml
   ```

   ### 11.2 Verify the app pods

   ```bash
   kubectl get pods -n default
   kubectl get svc -n default todo-app
   ```

   ### 11.3 Open the app

   The app should be reachable on:

   ```bash
   http://<HOST_IP>:30080
   http://<HOST_IP>:30080/admin
   ```

   ---

   ## 12. Train the AI model

   The AI service can run with a heuristic fallback, but for the full demo you should train the synthetic model.

   ### 12.1 Create a Python environment

   ```bash
   cd ai_service
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   ### 12.2 Train the model

   ```bash
   python train/train_model.py
   ```

   This creates:

   ```bash
   ai_service/model/niyojak_model.json
   ```

   If you are running the AI service from inside a container, rebuild and redeploy the AI service after the model is generated.

   ---

   ## 13. Run the AI service locally (optional)

   If you want to test the service outside Kubernetes first:

   ```bash
   cd ai_service/app
   python main.py
   ```

   Then open:

   - http://localhost:8000/health
   - http://localhost:8000/nodes

   ---

   ## 14. Run the full demo

   Once the cluster is healthy and the scheduler and AI service are running:

   1. Open the admin portal at:
      ```bash
      http://<HOST_IP>:30080/admin
      ```
   2. Choose a safe stress profile such as Light or Moderate.
   3. Click Stress Node on one of the worker nodes.
   4. Watch the node score drop.
   5. Trigger the traffic flood.
   6. Watch new pods get scheduled away from the stressed node.
   7. Release stress when you are done.

   ---

   ## 15. Useful debugging commands

   ```bash
   kubectl get nodes -o wide
   kubectl get pods -A
   kubectl describe pod -n default <pod-name>
   kubectl get events -A --sort-by=.metadata.creationTimestamp
   kubectl logs -n niyojak-system deploy/niyojak-aiservice --tail=100
   kubectl logs -n niyojak-system deploy/niyojak-scheduler --tail=100
   ```

   ---

   ## 16. Common issues

   ### The nodes do not join
   Check the k3s agent logs inside the VM:

   ```bash
   multipass exec niyojak-node2 -- sudo journalctl -u k3s-agent -n 100
   multipass exec niyojak-node3 -- sudo journalctl -u k3s-agent -n 100
   ```

   ### The scheduler pods are not running
   Check the deployment status:

   ```bash
   kubectl get deploy -n niyojak-system
   kubectl describe deploy -n niyojak-system niyojak-scheduler
   ```

   ### The app is not accessible
   Check the NodePort Service:

   ```bash
   kubectl get svc -n default todo-app
   ```

   ---

   ## 17. Recommended order for your presentation/demo

   Use this exact order:

   1. verify the cluster is healthy,
   2. verify the app is reachable,
   3. run a light stress on Node 2,
   4. trigger the flood,
   5. observe the scheduler moving new pods away from the stressed node,
   6. release the stress.

   That gives you a clean and safe live demonstration.
