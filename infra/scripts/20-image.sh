#!/usr/bin/env bash
# Prepare the base model, build the SuperExec app image, push it to the local
# registry, and pre-pull every image the demo uses into the kind nodes so the
# recording never waits on a download.
source "$(dirname "$0")/lib.sh"

log "Preparing the base model (fp32)"
"$ROOT/.venv/bin/python" "$ROOT/infra/scripts/prepare_model.py"

log "Building $APP_IMAGE"
# CUDA PyTorch when the branch clusters get the GPU, CPU-only otherwise.
if gpu_enabled; then default_index=$(torch_index); else default_index="https://download.pytorch.org/whl/cpu"; fi
info "PyTorch index: ${TORCH_INDEX:-$default_index}"
$CONTAINER_TOOL build -f "$ROOT/app/Containerfile" -t "$APP_IMAGE" \
  --build-arg TORCH_INDEX="${TORCH_INDEX:-$default_index}" \
  "$ROOT/models/granite-4.0-350m-fp32"

log "Pushing to the local registry"
if [ "$CONTAINER_TOOL" = "podman" ]; then
  podman push --tls-verify=false "$APP_IMAGE"
else
  docker push "$APP_IMAGE"
fi

prepull() {  # prepull <cluster> <image...>
  local n; n=$(node_of "$1"); shift
  for img in "$@"; do
    info "$n <- $img"
    $CONTAINER_TOOL exec "$n" crictl pull "$img" >/dev/null
  done
}

log "Pre-pulling images into the nodes"
prepull "$HUB" "$APP_IMAGE" "docker.io/flwr/superlink:$FLOWER_TAG" "$FL_CONTROLLER_IMAGE@$FL_CONTROLLER_DIGEST"
for c in $BRANCHES; do prepull "$c" "$APP_IMAGE" "docker.io/flwr/supernode:$FLOWER_TAG" "$DATA_SERVER_IMAGE"; done
# chile gets the SuperNode image too, so it is clear that only the Placement keeps it out.
prepull "$NO_DATA" "docker.io/flwr/supernode:$FLOWER_TAG"

if gpu_enabled; then
  log "Checking that pods in the branch clusters see the GPU"
  for c in $BRANCHES; do
    kubectl --context "kind-$c" delete pod gpu-check --ignore-not-found >/dev/null
    kubectl --context "kind-$c" run gpu-check --image="$APP_IMAGE" --image-pull-policy=IfNotPresent --restart=Never \
      --command -- python -c 'import torch; print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")' >/dev/null
    wait_for 300 "gpu-check pod in $c" sh -c "kubectl --context kind-$c get pod gpu-check -o jsonpath='{.status.phase}' | grep -Eq 'Succeeded|Failed'"
    result=$(kubectl --context "kind-$c" logs gpu-check 2>&1 | tail -1)
    kubectl --context "kind-$c" delete pod gpu-check --ignore-not-found >/dev/null
    case "$result" in
      "cuda True"*) info "$c: $result" ;;
      *) warn "$c: pods do not see the GPU ($result); training there will fall back to CPU" ;;
    esac
  done
fi
