#!/usr/bin/env bash
# Guided demo for the recording. Each step prints its command, waits for Enter,
# then runs it. AUTO=1 runs straight through (useful for a rehearsal).
# Start from a clean state with `make reset` (claims withdrawn, no FL resource).
source "$(dirname "$0")/../infra/scripts/lib.sh"
export PATH="$ROOT/.venv/bin:$PATH"   # flwr, python
set +e
cd "$ROOT"

step()  { printf '\n\033[1;35m━━━ %s\033[0m\n' "$*"; }
say()   { printf '\033[0;36m# %s\033[0m\n' "$*"; }
pause() { [ "${AUTO:-0}" = 1 ] || read -r -s _; }
run()   { printf '\033[1;32m$ %s\033[0m' "$*"; pause; echo; eval "$@"; }

phase() { kc -n "$FL_NAMESPACE" get federatedlearning "$FL_NAME" -o jsonpath='{.status.phase}' 2>/dev/null; }

# ----------------------------------------------------------------------------
step "1. The bank's fleet: one hub, four branch clusters"
say "Open Cluster Management hub + managed clusters (slide: OCM and FL share the hub-spoke model)"
run "kubectl --context kind-hub get managedclusters -L bank.demo/country"

step "2. Customer data lives only in each branch"
say "Brasil's messages are on the Brasil cluster's node, served only inside that cluster"
run "$CONTAINER_TOOL exec brasil-control-plane ls -l /data/bank"
run "kubectl --context kind-brasil -n bank-data get deploy,svc,networkpolicy"
say "Chile has a cluster and spare resources, but no customer-message dataset"
run "$CONTAINER_TOOL exec chile-control-plane ls /data/bank"

step "3. Platform pieces on the hub"
say "Flower SuperLink (aggregation endpoint, TLS) and the federated-learning-controller"
run "kubectl --context kind-hub -n flower-system get deploy,svc"
run "kubectl --context kind-hub -n open-cluster-management get deploy -l control-plane=controller-manager"
say "The flower-addon installs SuperNodes wherever its Placement says so: nowhere yet"
run "kubectl --context kind-hub get managedclusteraddons -A"

step "4. Create the FederatedLearning resource"
run "cat infra/manifests/hub/federatedlearning.yaml"
run "kubectl --context kind-hub apply -f infra/manifests/hub/federatedlearning.yaml"
sleep 5
run "kubectl --context kind-hub -n $FL_NAMESPACE get federatedlearning $FL_NAME -o wide"
say "Waiting: no cluster has declared the data yet, so the Placement selects nobody"

step "5. Branches declare that they hold the data (ClusterClaim)"
run "infra/scripts/61-publish-claims.sh"
say "The Placement now selects brasil, mexico and portugal; chile stays out"
say "The flower-addon installs a SuperNode (with the hub's CA certificate) in each selected branch"
until [ "$(phase)" = "Running" ]; do clear; demo/status.sh; sleep 3; done
clear; demo/status.sh
pause

step "6. Secure from registration"
say "Every klusterlet registered with a client certificate signed by the hub"
run "kubectl --context kind-hub get csr -o custom-columns='NAME:.metadata.name,REQUESTOR:.spec.username,SIGNER:.spec.signerName,STATUS:.status.conditions[0].type'"
say "The add-on delivered the CA to each branch; the SuperNode verifies the SuperLink with it"
run "kubectl --context kind-brasil -n flower-addon get secret flower-tls-ca"
run "kubectl --context kind-brasil -n flower-addon get deploy flower-supernode -o jsonpath='{.spec.template.spec.containers[0].args}' | tr ',' '\n'"

step "7. Train: the code goes to the data"
say "flwr run packages app/ as a Flower App Bundle and submits it to the SuperLink over TLS"
say "Tip: follow a branch in another terminal with: make logs BRANCH=brasil"
run "flwr run app ocm --stream"
# The log stream can end before the run does; wait until no run is still active.
until ! flwr ls ocm --format json 2>/dev/null | grep -q -E '"status": "(pending|starting|running)'; do sleep 5; done

step "8. Run finished; collect the global adapter from the hub"
run "flwr ls ocm"
run "infra/scripts/70-fetch-model.sh"

step "9. Same messages, three models"
if [ -f "$ROOT/models/adapters/brasil-only/adapter_model.safetensors" ]; then
  run "python eval/evaluate.py"
else
  say "models/adapters/brasil-only is missing; run 'make baseline' first for the Brazil-only column"
  run "python eval/evaluate.py"
fi
step "Done"
