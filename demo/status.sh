#!/usr/bin/env bash
# Compact status of the federation, as seen from the hub.
source "$(dirname "$0")/../infra/scripts/lib.sh"
set +e

b() { printf '\033[1m%s\033[0m\n' "$*"; }

b "Managed clusters (data claim | flower-addon)"
for c in $MANAGED; do
  claim=$(kc get managedcluster "$c" -o jsonpath="{.status.clusterClaims[?(@.name=='$DATA_CLAIM')].value}" 2>/dev/null)
  addon=$(kc get managedcluster "$c" -o jsonpath='{.metadata.labels.feature\.open-cluster-management\.io/addon-flower-addon}' 2>/dev/null)
  printf '  %-9s %-22s %s\n' "$c" "${claim:--}" "${addon:--}"
done

b "Placement decisions"
for p in "open-cluster-management/branches-with-customer-data" "$FL_NAMESPACE/$FL_NAME"; do
  ns=${p%%/*}; name=${p##*/}
  sel=$(kc -n "$ns" get placementdecisions -l "cluster.open-cluster-management.io/placement=$name" \
          -o jsonpath='{.items[*].status.decisions[*].clusterName}' 2>/dev/null)
  printf '  %-52s %s\n' "$p" "${sel:-(none)}"
done

b "FederatedLearning"
kc -n "$FL_NAMESPACE" get federatedlearning -o custom-columns='NAME:.metadata.name,PHASE:.status.phase,MESSAGE:.status.message' 2>/dev/null \
  | sed 's/^/  /' || echo "  (none)"

b "Flower pods"
kc -n flower-system get pods --no-headers 2>/dev/null | awk '{printf "  hub      %-55s %s\n", $1, $3}'
kc -n "$FL_NAMESPACE" get pods --no-headers 2>/dev/null | awk '{printf "  hub      %-55s %s\n", $1, $3}'
for c in $MANAGED; do
  kubectl --context "kind-$c" -n flower-addon get pods --no-headers 2>/dev/null | awk -v c="$c" '{printf "  %-8s %-55s %s\n", c, $1, $3}'
done
