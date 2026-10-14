#!/usr/bin/env bash
# Check the tools the demo needs, fetch clusteradm and the pinned upstream
# sources, and create the Python environment. Safe to re-run.
source "$(dirname "$0")/lib.sh"

OS=$(uname -s)

log "Checking tools ($OS, container tool: $CONTAINER_TOOL)"
missing=""
for t in "$CONTAINER_TOOL" kind kubectl helm uv openssl git curl; do
  if command -v "$t" >/dev/null 2>&1; then info "found $t"; else missing="$missing $t"; fi
done
if [ -n "$missing" ]; then
  if [ "$OS" = "Darwin" ]; then
    die "missing:$missing  ->  brew install podman kind kubernetes-cli helm uv"
  else
    die "missing:$missing  ->  see README.md, 'Running on a Linux server'"
  fi
fi

# Re-download if missing or built for another OS/arch (e.g. the folder was copied from a Mac).
if ! "$ROOT/bin/clusteradm" version >/dev/null 2>&1; then
  log "Downloading clusteradm v$OCM_BUNDLE_VERSION into bin/"
  os=$(echo "$OS" | tr '[:upper:]' '[:lower:]'); arch=$(uname -m)
  case "$arch" in x86_64) arch=amd64 ;; aarch64) arch=arm64 ;; esac
  mkdir -p "$ROOT/bin"
  curl -sSL "https://github.com/open-cluster-management-io/clusteradm/releases/download/v${OCM_BUNDLE_VERSION}/clusteradm_${os}_${arch}.tar.gz" \
    | tar -xz -C "$ROOT/bin" clusteradm
fi
info "$(clusteradm version 2>/dev/null | head -1)"

log "Checking the container runtime"
if [ "$CONTAINER_TOOL" = "podman" ]; then
  podman info --format '{{.Host.Security.Rootless}}' 2>/dev/null | grep -q false || {
    if [ "$OS" = "Darwin" ]; then
      die "start a rootful Podman machine: podman machine init --cpus 8 --memory 12288 --disk-size 120 --rootful && podman machine start"
    else
      die "kind needs rootful Podman on Linux: run with sudo (sudo -E make ...), or use CONTAINER_TOOL=docker"
    fi
  }
  mem=$(podman info --format '{{.Host.MemTotal}}'); cpus=$(podman info --format '{{.Host.CPUs}}')
else
  docker info >/dev/null 2>&1 || die "docker is not running (or this user cannot reach it)"
  mem=$(docker info --format '{{.MemTotal}}'); cpus=$(docker info --format '{{.NCPU}}')
fi
info "runtime sees ${cpus} CPUs, $(( mem / 1024 / 1024 / 1024 )) GiB RAM"
[ "$mem" -ge 11000000000 ] || warn "less than ~11 GiB for containers; three branches training at once may run out of memory"
[ "$cpus" -ge 8 ] || warn "fewer than 8 CPUs; five clusters plus training will be slow"

if [ "$OS" = "Linux" ]; then
  # Five kind clusters exceed the default inotify limits on many distributions.
  w=$(sysctl -n fs.inotify.max_user_watches 2>/dev/null || echo 0)
  i=$(sysctl -n fs.inotify.max_user_instances 2>/dev/null || echo 0)
  if [ "$w" -lt 524288 ] || [ "$i" -lt 512 ]; then
    warn "raise inotify limits: sudo sysctl fs.inotify.max_user_watches=524288 fs.inotify.max_user_instances=512"
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  info "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
fi
if gpu_enabled; then
  log "Checking that kind nodes can receive the GPU"
  [ "$CONTAINER_TOOL" = "docker" ] || die "GPU inside the kind clusters needs Docker (CONTAINER_TOOL=docker), or set GPU=0"
  runtime=$(docker info --format '{{.DefaultRuntime}}' 2>/dev/null || true)
  if [ "$runtime" != "nvidia" ] || ! grep -Eq '^[[:space:]]*accept-nvidia-visible-devices-as-volume-mounts[[:space:]]*=[[:space:]]*true' /etc/nvidia-container-runtime/config.toml 2>/dev/null; then
    die "Docker is not set up to hand the GPU to kind nodes. Run once: sudo infra/scripts/05-gpu-host.sh  (or GPU=0 to train on CPU)"
  fi
  info "Docker default runtime: nvidia; volume-mount GPU requests: enabled"
  info "branch clusters ($BRANCHES) will train on the GPU"
else
  info "GPU in the clusters: off (training in the clusters runs on CPU)"
fi

if [ ! -d "$UPSTREAM_DIR/.git" ]; then
  log "Fetching open-cluster-management-io/addon-contrib @ ${UPSTREAM_REF:0:10}"
  mkdir -p "$(dirname "$UPSTREAM_DIR")"
  git clone -q --filter=blob:none --no-checkout "$UPSTREAM_REPO" "$UPSTREAM_DIR"
  git -C "$UPSTREAM_DIR" sparse-checkout set federated-learning-controller flower-addon
  git -C "$UPSTREAM_DIR" checkout -q "$UPSTREAM_REF"
fi
info "upstream: $(git -C "$UPSTREAM_DIR" log -1 --format='%h %cd %s' --date=short)"

if ! "$ROOT/.venv/bin/python" -c 'import sys' >/dev/null 2>&1; then
  log "Creating the Python environment (.venv)"
  rm -rf "$ROOT/.venv"
  uv venv "$ROOT/.venv" --python 3.12
fi
if [ "$OS" = "Linux" ]; then
  # Pick the PyTorch build that matches the driver (CUDA 12.6 or 13.0) or CPU.
  uv pip install -q --python "$ROOT/.venv/bin/python" --index-url "$(torch_index)" "torch==2.14.1"
fi
uv pip install -q --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements.txt"
info "python env: $("$ROOT/.venv/bin/python" -c 'import flwr, torch; d = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"); print("flwr", flwr.__version__, "| torch", torch.__version__, "| device", d)')"

write_flwr_config
