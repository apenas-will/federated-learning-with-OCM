#!/usr/bin/env bash
# Bring the federation back to the state the demo starts from:
# no FederatedLearning resource, no data claims, so no SuperNodes in the branches.
source "$(dirname "$0")/../infra/scripts/lib.sh"

log "Deleting the FederatedLearning resource"
kc delete -f "$ROOT/infra/manifests/hub/federatedlearning.yaml" --ignore-not-found --wait=true

log "Withdrawing the data claims"
"$ROOT/infra/scripts/61-publish-claims.sh" --remove

log "Waiting for the flower-addon to be removed from the branches"
wait_for 180 "add-ons removed" sh -c "[ -z \"\$(kubectl --context $HUB_CTX get managedclusteraddons -A --no-headers 2>/dev/null | grep flower-addon)\" ]"
for c in $MANAGED; do
  wait_for 120 "flower-addon namespace gone in $c" sh -c "! kubectl --context kind-$c get ns flower-addon"
done
info "clean"
