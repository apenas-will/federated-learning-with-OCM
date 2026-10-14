#!/usr/bin/env bash
# Put each branch's dataset on its own cluster and serve it there.
# Data for brasil goes only to the brasil cluster, and so on; chile gets none.
# This step does NOT publish the ClusterClaim yet: that is part of the demo
# (scripts/demo.sh, `make publish-data`).
source "$(dirname "$0")/lib.sh"

for b in $BRANCHES; do
  n=$(node_of "$b")
  log "Branch $b: copying data/clients/$b into its node ($n:/data/bank)"
  $CONTAINER_TOOL exec "$n" mkdir -p /data/bank
  for f in train.jsonl val.jsonl; do
    $CONTAINER_TOOL cp "$ROOT/data/clients/$b/$f" "$n:/data/bank/$f"
  done
  $CONTAINER_TOOL exec "$n" sh -c 'chmod 0755 /data/bank && chmod 0644 /data/bank/*.jsonl && wc -l /data/bank/*.jsonl'

  kubectl --context "kind-$b" apply -f "$ROOT/infra/manifests/branch/branch-data.yaml" >/dev/null
  kubectl --context "kind-$b" -n bank-data rollout status deploy/branch-data --timeout=120s
done
