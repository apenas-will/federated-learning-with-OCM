#!/usr/bin/env bash
# One command, no prompts: set up everything that is missing, run the whole
# federated training through OCM, and collect the metrics and evidence in results/.
#
#   demo/run-all.sh                 # or: make run-all
#   THREADS=8 demo/run-all.sh       # CPU threads per branch (default: CPUs/4, 2..8)
#   SKIP_SETUP=1 demo/run-all.sh    # clusters already up
source "$(dirname "$0")/../infra/scripts/lib.sh"
cd "$ROOT"

R="$ROOT/results"
mkdir -p "$R/logs" "$R/cluster"
TIMINGS="$R/timings.tsv"
printf 'phase\tseconds\n' > "$TIMINGS"
phase_start=0
begin() { log "$1"; phase_name="$1"; phase_start=$(date +%s); }
end()   { printf '%s\t%s\n' "$phase_name" "$(( $(date +%s) - phase_start ))" >> "$TIMINGS"; }

ncpu=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)
THREADS="${THREADS:-$(( ncpu / 4 ))}"
[ "$THREADS" -lt 2 ] && THREADS=2; [ "$THREADS" -gt 8 ] && THREADS=8

# --- 0. setup ---------------------------------------------------------------
if [ -z "${SKIP_SETUP:-}" ]; then
  for s in 00-check-prereqs 10-clusters 20-image 30-ocm 40-flower-addon 50-fl-controller 60-branch-data; do
    begin "setup: $s"; "infra/scripts/$s.sh"; end
  done
fi

# --- 1. clean starting point ---------------------------------------------------
begin "reset"; demo/reset.sh; end

# --- 2. FederatedLearning resource -> Waiting -----------------------------------
begin "create FederatedLearning (expect Waiting)"
kc apply -f infra/manifests/hub/federatedlearning.yaml
wait_for 120 "Waiting phase" sh -c "kubectl --context $HUB_CTX -n $FL_NAMESPACE get federatedlearning $FL_NAME -o jsonpath='{.status.phase}' | grep -q Waiting"
sleep 5
demo/status.sh > "$R/cluster/1-waiting.txt" 2>&1
end

# --- 3. branches publish their data claims -> Placement -> Running --------------
begin "publish data claims (expect Running)"
infra/scripts/61-publish-claims.sh
wait_for 900 "Running phase" sh -c "kubectl --context $HUB_CTX -n $FL_NAMESPACE get federatedlearning $FL_NAME -o jsonpath='{.status.phase}' | grep -q Running"
for b in $BRANCHES; do
  # The controller labels the ClientApp pods but not their Deployment: wait on it by name.
  wait_for 600 "ClientApp ready in $b" kubectl --context "kind-$b" -n flower-addon wait --for=condition=Available \
    "deploy/$FL_NAME-clientapp" --timeout=10s
done
kc -n "$FL_NAMESPACE" wait --for=condition=Available deploy -l app.kubernetes.io/component=superexec-serverapp --timeout=300s
demo/status.sh > "$R/cluster/2-running.txt" 2>&1
end

begin "collect cluster evidence"
{
  echo "## managed clusters"; kc get managedclusters -L bank.demo/country
  echo; echo "## placements"; kc get placements -A
  echo; echo "## placement decisions"
  kc get placementdecisions -A -o custom-columns='NAMESPACE:.metadata.namespace,PLACEMENT:.metadata.labels.cluster\.open-cluster-management\.io/placement,CLUSTERS:.status.decisions[*].clusterName'
  echo; echo "## add-ons"; kc get managedclusteraddons -A
  echo; echo "## FederatedLearning"; kc -n "$FL_NAMESPACE" get federatedlearning "$FL_NAME" -o yaml
  echo; echo "## ManifestWorkReplicaSet"; kc -n "$FL_NAMESPACE" get manifestworkreplicasets
  echo; echo "## registration certificates (CSRs)"
  kc get csr -o custom-columns='NAME:.metadata.name,REQUESTOR:.spec.username,SIGNER:.spec.signerName,STATUS:.status.conditions[0].type'
  echo; echo "## SuperNode args in brasil (TLS with the CA shipped by the add-on)"
  kubectl --context kind-brasil -n flower-addon get deploy flower-supernode -o jsonpath='{.spec.template.spec.containers[0].args}'; echo
} > "$R/cluster/3-evidence.txt" 2>&1
end

# --- 4. training --------------------------------------------------------------
begin "federated training (flwr run, $THREADS CPU threads per branch)"
flwr_run_and_wait ocm "$R/logs/flwr-run.log" "num-threads=$THREADS"
"$ROOT/.venv/bin/flwr" ls ocm > "$R/logs/flwr-runs.txt" 2>&1 || true
end

begin "collect logs"
kc -n "$FL_NAMESPACE" logs -l app.kubernetes.io/component=superexec-serverapp --tail=-1 > "$R/logs/hub-serverapp.log" 2>&1 || true
for b in $BRANCHES; do
  kubectl --context "kind-$b" -n flower-addon logs -l app.kubernetes.io/component=superexec-clientapp --tail=-1 > "$R/logs/$b-clientapp.log" 2>&1 || true
  kubectl --context "kind-$b" -n bank-data logs deploy/branch-data --tail=-1 > "$R/logs/$b-data-access.log" 2>&1 || true
done
end

begin "fetch the federated adapter"; infra/scripts/70-fetch-model.sh; cp models/adapters/federated/history.json "$R/history-federated.json"; end

# --- 5. evaluation (GPU if available) -----------------------------------------
if [ ! -f models/adapters/brasil-only/adapter_model.safetensors ]; then
  begin "train the brasil-only baseline"; "$ROOT/.venv/bin/python" eval/train_local.py --branch brasil; end
fi
begin "evaluate base / brasil-only / federated"; "$ROOT/.venv/bin/python" eval/evaluate.py; end

log "Done. Metrics and evidence:"
info "results/summary.md          accuracy tables + slide messages + candidate messages"
info "results/evaluation.json     full numbers"
info "results/history-federated.json  per-round, per-branch metrics"
info "results/timings.tsv         duration of each phase"
info "results/cluster/            Waiting/Running snapshots, placements, add-ons, CSRs"
info "results/logs/               flwr run, ServerApp, ClientApps, data-service access logs"
column -t -s $'\t' "$TIMINGS" 2>/dev/null || cat "$TIMINGS"
