#!/usr/bin/env bash
# Deploy the upstream flower-addon (unmodified chart, pinned commit):
#   - SuperLink on the hub, with TLS on the Fleet API (SuperNodes) and Control API (flwr CLI)
#   - an OCM add-on that installs a SuperNode in each selected branch cluster and
#     ships it the CA certificate, so SuperNodes can verify the SuperLink
# Then switch the add-on from manual install to Placement-driven install: only
# branches that hold customer data (and have spare resources) get a SuperNode.
source "$(dirname "$0")/lib.sh"

CHART="$UPSTREAM_DIR/flower-addon/charts/flower-addon"
HUB_IP=$(hub_ip)

log "TLS certificates for the SuperLink (SANs: superlink service, hub node $HUB_IP, 127.0.0.1)"
if ! kc -n flower-system get secret flower-superlink-tls >/dev/null 2>&1; then
  # The upstream script uses the current kubectl context.
  kubectl config use-context "$HUB_CTX" >/dev/null
  "$UPSTREAM_DIR/flower-addon/hack/generate-certs.sh" --hub-ip "$HUB_IP" --hub-ip 127.0.0.1
fi
mkdir -p "$FLWR_HOME"
kc -n flower-system get secret flower-tls-ca -o jsonpath='{.data.ca\.crt}' | base64 -d > "$FLWR_HOME/ca.crt"
write_flwr_config
info "CA certificate for the flwr CLI written to .flwr/ca.crt (connection 'ocm' in .flwr/config.toml)"

log "Installing the flower-addon chart (Flower $FLOWER_VERSION)"
# generate-certs.sh creates flower-system, which the chart also templates;
# mark it as owned by this release so Helm adopts it instead of refusing.
kc label namespace flower-system app.kubernetes.io/managed-by=Helm --overwrite >/dev/null
kc annotate namespace flower-system meta.helm.sh/release-name=flower-addon \
  meta.helm.sh/release-namespace=default --overwrite >/dev/null
helm upgrade --install flower-addon "$CHART" --kube-context "$HUB_CTX" \
  --set tls.enabled=true \
  --set deploymentConfig.superlinkAddress="$HUB_IP" \
  --set superlink.image.tag="$FLOWER_TAG" \
  --set supernode.image.tag="$FLOWER_TAG" \
  --set supernode.numPartitions=3
kc -n flower-system rollout status deploy/superlink --timeout=180s

log "Placement-driven install: SuperNodes only where the data is"
placement="$ROOT/infra/manifests/hub/flower-addon-placement.yaml"
# The resource criterion is a CEL expression on the cluster's allocatable CPU.
# Check it on a throwaway Placement first; if this OCM version cannot evaluate
# it, fall back to the data-locality criterion alone instead of selecting nobody.
cel_ok=false
cat <<EOF | kc apply -f - >/dev/null
apiVersion: cluster.open-cluster-management.io/v1beta1
kind: Placement
metadata:
  name: cel-selftest
  namespace: open-cluster-management
spec:
  clusterSets: [global]
  predicates:
    - requiredClusterSelector:
$(sed -n '/# cel-begin/,/# cel-end/p' "$placement" | grep -v '# cel-')
EOF
for _ in $(seq 1 20); do
  n=$(kc -n open-cluster-management get placement cel-selftest -o jsonpath='{.status.numberOfSelectedClusters}' 2>/dev/null || echo 0)
  [ "${n:-0}" -ge 4 ] && { cel_ok=true; break; }
  sleep 3
done
kc -n open-cluster-management delete placement cel-selftest --ignore-not-found >/dev/null
if $cel_ok; then
  info "CEL resource check works: every cluster has >= 2 allocatable CPUs"
  kc apply -f "$placement"
else
  warn "CEL resource check selected no clusters on this OCM version; using the data-locality criterion only"
  sed '/# cel-begin/,/# cel-end/d' "$placement" | kc apply -f -
fi
kc patch clustermanagementaddon flower-addon --type=merge -p '{
  "spec": {"installStrategy": {"type": "Placements", "placements": [
    {"name": "branches-with-customer-data", "namespace": "open-cluster-management"}]}}}'
kc get clustermanagementaddon flower-addon -o jsonpath='{.spec.installStrategy}'; echo
