#!/usr/bin/env bash
# Create a local image registry and five kind clusters:
#   hub                    - OCM hub, FL controller, Flower SuperLink (aggregator)
#   brasil mexico portugal - branch clusters that own customer-message data
#   chile                  - a branch cluster with resources but no data
source "$(dirname "$0")/lib.sh"

log "Local image registry ($REGISTRY_NAME on 127.0.0.1:$REGISTRY_PORT)"
if [ "$($CONTAINER_TOOL inspect -f '{{.State.Running}}' "$REGISTRY_NAME" 2>/dev/null || true)" != "true" ]; then
  $CONTAINER_TOOL run -d --restart=always -p "127.0.0.1:${REGISTRY_PORT}:5000" --name "$REGISTRY_NAME" docker.io/library/registry:2 >/dev/null
fi

create_cluster() {  # create_cluster <name> [extra yaml appended to the node]
  local name="$1" extra="${2:-}"
  if kind get clusters 2>/dev/null | grep -qx "$name"; then info "cluster $name exists"; return; fi
  log "Creating kind cluster $name"
  cat <<EOF | kind create cluster --name "$name" --wait 120s --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
containerdConfigPatches:
- |-
  [plugins."io.containerd.cri.v1.images".registry]
    config_path = "/etc/containerd/certs.d"
nodes:
- role: control-plane
$extra
EOF
}

# The hub exposes the Flower Control API (NodePort 30093) to the laptop for `flwr run`.
create_cluster "$HUB" "  extraPortMappings:
  - containerPort: 30093
    hostPort: 30093
    listenAddress: 127.0.0.1
    protocol: TCP"

# With GPU on, each data branch's node asks the host's NVIDIA runtime for the
# GPU through this volume mount (the mechanism NVIDIA's nvkind uses). The three
# branches share the same physical GPU; chile and the hub get none.
GPU_MOUNT="  extraMounts:
  - hostPath: /dev/null
    containerPath: /var/run/nvidia-container-devices/all"
for c in $MANAGED; do
  if gpu_enabled && echo " $BRANCHES " | grep -q " $c "; then create_cluster "$c" "$GPU_MOUNT"; else create_cluster "$c"; fi
done

setup_gpu_node() {  # NVIDIA Container Toolkit inside the kind node, as the node's default runtime
  local n; n=$(node_of "$1")
  if $CONTAINER_TOOL exec "$n" sh -c 'command -v nvidia-ctk' >/dev/null 2>&1; then info "$n: GPU runtime already set up"; return; fi
  log "GPU runtime in $n"
  $CONTAINER_TOOL exec "$n" nvidia-smi -L || die "$n does not see the GPU; run: sudo infra/scripts/05-gpu-host.sh"
  $CONTAINER_TOOL exec "$n" bash -c '
    set -e
    apt-get update -qq
    apt-get install -y -qq gpg curl >/dev/null
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y -qq nvidia-container-toolkit >/dev/null
    # Default runtime, so pods get the GPU without a RuntimeClass: the
    # federated-learning-controller creates the ClientApp pods and sets neither
    # a RuntimeClass nor GPU requests. The app image sets NVIDIA_VISIBLE_DEVICES=all.
    nvidia-ctk runtime configure --runtime=containerd --config-source=file --set-as-default
    nvidia-ctk config --set accept-nvidia-visible-devices-envvar-when-unprivileged=true --in-place
    systemctl restart containerd
  '
  wait_for 120 "containerd back in $n" $CONTAINER_TOOL exec "$n" crictl info
}

log "Connecting the registry to the kind network and pointing every node at it"
$CONTAINER_TOOL network connect kind "$REGISTRY_NAME" 2>/dev/null || true
for c in $HUB $MANAGED; do
  n=$(node_of "$c")
  $CONTAINER_TOOL exec "$n" mkdir -p "/etc/containerd/certs.d/localhost:${REGISTRY_PORT}"
  printf '[host."http://%s:5000"]\n' "$REGISTRY_NAME" | $CONTAINER_TOOL exec -i "$n" cp /dev/stdin "/etc/containerd/certs.d/localhost:${REGISTRY_PORT}/hosts.toml"
done

if gpu_enabled; then
  for c in $BRANCHES; do setup_gpu_node "$c"; done
  for c in $BRANCHES; do
    kubectl --context "kind-$c" wait --for=condition=Ready node --all --timeout=180s >/dev/null
  done
fi

log "Clusters"
kind get clusters
