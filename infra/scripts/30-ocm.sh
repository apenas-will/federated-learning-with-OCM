#!/usr/bin/env bash
# Install Open Cluster Management: hub on `hub`, klusterlet on every branch
# cluster, then accept their registration. Registration is certificate based:
# each klusterlet gets a client certificate signed by the hub (see `kubectl get csr`).
source "$(dirname "$0")/lib.sh"

log "Initializing the OCM hub (bundle $OCM_BUNDLE_VERSION)"
if ! kc get clustermanager cluster-manager >/dev/null 2>&1; then
  clusteradm init --wait --context "$HUB_CTX" --bundle-version="$OCM_BUNDLE_VERSION" \
    --feature-gates=ManifestWorkReplicaSet=true
fi
# The FL controller distributes ClientApps with a ManifestWorkReplicaSet (alpha, off by default).
kc patch clustermanager cluster-manager --type=merge \
  -p '{"spec":{"workConfiguration":{"featureGates":[{"feature":"ManifestWorkReplicaSet","mode":"Enable"}]}}}'

joincmd=$(clusteradm get token --context "$HUB_CTX" | grep '^clusteradm join')
for c in $MANAGED; do
  if kc get managedcluster "$c" >/dev/null 2>&1; then info "$c already registered"; continue; fi
  log "Joining $c to the hub"
  # --force-internal-endpoint-lookup: reach the hub API server over the kind network.
  # On a busy machine the klusterlet can outlive clusteradm's wait; keep going and
  # wait for the registration request below instead of failing here.
  eval "${joincmd//<cluster_name>/$c} --bundle-version=$OCM_BUNDLE_VERSION --force-internal-endpoint-lookup --wait --context kind-$c" \
    || warn "clusteradm join $c did not confirm in time; waiting for its registration request"
  wait_for 600 "registration request from $c" sh -c "kubectl --context $HUB_CTX get csr -l open-cluster-management.io/cluster-name=$c -o name | grep -q ."
done

log "Accepting the branch clusters"
clusteradm accept --clusters "$(echo $MANAGED | tr ' ' ',')" --wait --context "$HUB_CTX"

log "Labelling clusters with their country"
kc label managedcluster brasil  bank.demo/country=BR --overwrite
kc label managedcluster mexico  bank.demo/country=MX --overwrite
kc label managedcluster portugal bank.demo/country=PT --overwrite
kc label managedcluster chile   bank.demo/country=CL --overwrite

log "Namespaces and ClusterSet bindings on the hub"
kc apply -f "$ROOT/infra/manifests/hub/namespaces-and-bindings.yaml"

wait_for 300 "work controller with ManifestWorkReplicaSet" \
  sh -c "kubectl --context $HUB_CTX get crd manifestworkreplicasets.work.open-cluster-management.io"
kc get managedclusters
