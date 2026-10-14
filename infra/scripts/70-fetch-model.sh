#!/usr/bin/env bash
# Copy the federated LoRA adapter (and the round-by-round metrics) out of the
# SuperExec-ServerApp pod on the hub into models/adapters/federated.
# The pod has no volume, so do this before deleting the FederatedLearning resource.
source "$(dirname "$0")/lib.sh"

pod=$(kc -n "$FL_NAMESPACE" get pods -l "app.kubernetes.io/component=superexec-serverapp" \
        -o jsonpath='{.items[0].metadata.name}')
[ -n "$pod" ] || die "no ServerApp pod in $FL_NAMESPACE; is the FederatedLearning resource deployed?"

run_dir=$(kc -n "$FL_NAMESPACE" exec "$pod" -- sh -c 'readlink /app/output/latest' 2>/dev/null || true)
[ -n "$run_dir" ] || die "no finished run found in $pod:/app/output (did 'flwr run' complete?)"

out="$ROOT/models/adapters/federated"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
log "Copying $pod:/app/output/$run_dir"
kc -n "$FL_NAMESPACE" cp "$pod:/app/output/$run_dir" "$tmp/run" >/dev/null
[ -f "$tmp/run/adapter/adapter_model.safetensors" ] || die "adapter not found in the copied run directory"
rm -rf "$out"; mkdir -p "$out"
cp -R "$tmp/run/adapter/." "$out/"
cp "$tmp/run/history.json" "$out/history.json"
ls -la "$out"
