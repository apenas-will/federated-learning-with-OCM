#!/usr/bin/env bash
# Each branch that holds customer messages publishes a ClusterClaim on its own
# cluster. The klusterlet reports it to the hub, where Placements can select on
# it. The claim says "this cluster has the dataset", never the data itself.
#   61-publish-claims.sh            publish for every data branch
#   61-publish-claims.sh --remove   withdraw the claims
source "$(dirname "$0")/lib.sh"

for b in $BRANCHES; do
  if [ "${1:-}" = "--remove" ]; then
    kubectl --context "kind-$b" delete clusterclaim "$DATA_CLAIM" --ignore-not-found
    continue
  fi
  cat <<EOF | kubectl --context "kind-$b" apply -f -
apiVersion: cluster.open-cluster-management.io/v1alpha1
kind: ClusterClaim
metadata:
  name: $DATA_CLAIM
spec:
  value: banking77-$b-v1
EOF
done

if [ "${1:-}" != "--remove" ]; then
  log "Waiting for the hub to see the claims"
  for b in $BRANCHES; do
    wait_for 120 "claim from $b" sh -c \
      "kubectl --context $HUB_CTX get managedcluster $b -o jsonpath='{.status.clusterClaims[*].name}' | grep -q $DATA_CLAIM"
    info "$b: $(kc get managedcluster "$b" -o jsonpath="{.status.clusterClaims[?(@.name=='$DATA_CLAIM')].value}")"
  done
fi
