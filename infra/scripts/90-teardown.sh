#!/usr/bin/env bash
# Delete the five kind clusters and the local registry. Data, models and
# adapters on the laptop are kept.
source "$(dirname "$0")/lib.sh"

for c in $HUB $MANAGED; do
  if kind get clusters 2>/dev/null | grep -qx "$c"; then
    log "Deleting cluster $c"
    kind delete cluster --name "$c"
  fi
done
log "Removing the local registry"
$CONTAINER_TOOL rm -f "$REGISTRY_NAME" >/dev/null 2>&1 || true
