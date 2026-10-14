#!/usr/bin/env bash
# Run the same Flower app as the cluster demo, but as local processes:
# one SuperLink, three SuperNodes (brasil, mexico, portugal) and one small data
# server per branch. No Kubernetes, no OCM. Uses the GPU when PyTorch sees one.
#
# Useful to collect metrics quickly on a GPU server or to iterate on the app.
# The federated adapter ends up in models/adapters/federated-sim.
#
#   infra/scripts/local-federation.sh                      # defaults from app/pyproject.toml
#   infra/scripts/local-federation.sh 'num-server-rounds=10'
source "$(dirname "$0")/lib.sh"

RUN_CONFIG="${1:-}"
PY="$ROOT/.venv/bin"
LOGS="$ROOT/results/local-federation"
OUT="$ROOT/models/fl-output-local"
FLEET_PORT=19092; SERVERAPPIO_PORT=19091
pids=""

cleanup() {
  for p in $pids; do kill "$p" 2>/dev/null || true; done
  # SuperExec children started by the SuperNodes/SuperLink
  pkill -f "flower-superexec.*127.0.0.1:19(09|1)" 2>/dev/null || true
  sleep 3
  for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
  pkill -9 -f "flower-superexec.*127.0.0.1:19(09|1)" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Only Python is needed here (no clusters): create .venv if it is missing or foreign.
if ! "$PY/python" -c 'import flwr, torch, peft' >/dev/null 2>&1; then
  log "Creating the Python environment (.venv)"
  rm -rf "$ROOT/.venv"; uv venv "$ROOT/.venv" --python 3.12
  [ "$(uname -s)" = "Linux" ] && uv pip install -q --python "$PY/python" --index-url "$(torch_index)" "torch==2.14.1"
  uv pip install -q --python "$PY/python" -r "$ROOT/requirements.txt"
fi
[ -f "$ROOT/models/granite-4.0-350m-fp32/model.safetensors" ] || "$PY/python" "$ROOT/infra/scripts/prepare_model.py"
write_flwr_config
rm -rf "$LOGS"; mkdir -p "$LOGS" "$OUT"
export BASE_MODEL_PATH="$ROOT/models/granite-4.0-350m-fp32" FL_OUTPUT_DIR="$OUT"
export PATH="$PY:$PATH"   # the SuperLink/SuperNodes start `flower-superexec` from PATH

log "Starting the data servers (one per branch, localhost only)"
i=0
for b in $BRANCHES; do
  i=$((i + 1))
  "$PY/python" -m http.server "1810$i" --bind 127.0.0.1 --directory "$ROOT/data/clients/$b" > "$LOGS/data-$b.log" 2>&1 &
  pids="$pids $!"
done

log "Starting the SuperLink"
# Use .venv as is (on a Linux GPU server it has CUDA PyTorch) instead of a per-run `uv sync` env.
"$PY/flower-superlink" --insecure --disable-runtime-dependency-installation \
  --serverappio-api-address "127.0.0.1:$SERVERAPPIO_PORT" \
  --fleet-api-address "127.0.0.1:$FLEET_PORT" \
  --control-api-address "127.0.0.1:$LOCAL_CONTROL_PORT" > "$LOGS/superlink.log" 2>&1 &
pids="$pids $!"
sleep 5

log "Starting one SuperNode per branch"
i=0
for b in $BRANCHES; do
  i=$((i + 1))
  "$PY/flower-supernode" --insecure --superlink "127.0.0.1:$FLEET_PORT" \
    --clientappio-api-address "127.0.0.1:1910$i" \
    --node-config "cluster-name=\"$b\" data-endpoint=\"http://127.0.0.1:1810$i\"" > "$LOGS/supernode-$b.log" 2>&1 &
  pids="$pids $!"
done
sleep 5

device=$("$PY/python" -c 'import sys; sys.path.insert(0, sys.argv[1]); from bank_router.task import pick_device; print(pick_device())' "$ROOT/app")
log "Running the federation (device: $device)"
flwr_run_and_wait local "$LOGS/flwr-run.log" "$RUN_CONFIG"

for b in $BRANCHES; do grep -h "^\[$b\]" "$LOGS/supernode-$b.log" > "$LOGS/clientapp-$b.log" || true; done

[ -L "$OUT/latest" ] || die "the run did not produce an adapter; see $LOGS"
dest="$ROOT/models/adapters/federated-sim"
rm -rf "$dest"; mkdir -p "$dest"
cp -R "$OUT/latest/adapter/." "$dest/"
cp "$OUT/latest/history.json" "$dest/history.json"
log "Federated adapter (local processes) saved to models/adapters/federated-sim; logs in results/local-federation"
