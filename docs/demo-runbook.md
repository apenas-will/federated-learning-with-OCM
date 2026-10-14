# Demo runbook

How to record the 4–5 minute demo video (slide 21) and fill in the results
slide (slide 22).

## Before recording

```bash
make setup        # clusters, OCM, flower-addon, controller, branch data (once)
make baseline     # "trained only in Brazil" adapter, used in the comparison
make reset        # starting point: no FederatedLearning resource, no data claims
```

Optional rehearsal: `AUTO=1 make demo` runs every step without pausing, then
`make reset` again.

Terminal layout that works well on video (three panes):

| Pane | Command |
|---|---|
| main | `make demo` |
| right top | `make watch` (clusters, claims, Placement decisions, FL phase, pods; refreshes every 3 s) |
| right bottom | `make logs BRANCH=brasil` once step 5 is done (or `make data-access BRANCH=brasil`) |

Federated training takes several minutes on CPU; speed it up in editing.

## Steps

`make demo` (`demo/demo.sh`) prints each command and waits for Enter.

| # | What you show | Command (run by the script) | Slide | Say |
|---|---|---|---|---|
| 1 | Hub and four branch clusters | `kubectl get managedclusters -L bank.demo/country` | 13 | OCM hub-spoke = FL aggregator-collaborators |
| 2 | Data only in each branch; Chile has none | `ls /data/bank` on the brasil and chile nodes; `branch-data` service + NetworkPolicy | 4, 5 | The messages never leave the country; only a local service can read them |
| 3 | Platform on the hub: SuperLink (TLS), FL controller; no add-on installed yet | `kubectl -n flower-system get deploy,svc`, `get managedclusteraddons -A` | 17 (step 0) | Prerequisites: add-on registered, app packaged as a container |
| 4 | Create the `FederatedLearning` resource; phase **Waiting** | `kubectl apply -f infra/manifests/hub/federatedlearning.yaml` | 17 (steps 1–2) | Framework, image, minimum of 3 participants, Placement |
| 5 | Branches publish their data ClusterClaim; Placement selects brasil, mexico, portugal and leaves chile out; SuperNodes appear; phase **Running** | `infra/scripts/61-publish-claims.sh`, then a live status view | 15, 17 (steps 3–4) | Data locality + resources decide who participates |
| 6 | Certificates | `kubectl get csr`, the CA Secret in brasil, SuperNode `--root-certificates` | 16 | Registration is certificate based; the add-on ships the CA used for TLS |
| 7 | Training: the code goes to the data | `flwr run app ocm --stream` | 8, 9, 17 (steps 5–6) | Each round: 5.7 MB adapter out, local training, 5.7 MB back, FedAvg; base model and data stay put |
| 8 | Run finished; collect the adapter | `flwr ls ocm`, `infra/scripts/70-fetch-model.sh` | 17 (step 7) | The run status replaces the old "Completed" phase |
| 9 | Same messages, three models | `python eval/evaluate.py` | 20, 22 | Base routes everything to `cards`; Brazil-only is weaker on cards and fx_fees; federated covers all queues |

What to point at in the training logs:

- `[hub] round N: received 5.7 MB LoRA update from brasil` (only the adapter travels)
- `[brasil] round N: 1777 local messages loaded from http://branch-data.bank-data.svc...` (local read)
- `[hub] round N: local validation accuracy brasil … | mexico … | portugal …` (the hub only sees numbers)

## Slide updates for the current upstream controller

The slides were written for the controller described in the January 2026 blog.
Since March 2026 the Flower path works differently:

| Slide | Was | Now |
|---|---|---|
| 16 | The collaborator is an add-on and receives certificates when it registers | The **flower-addon** installs the Flower SuperNode in each selected cluster and ships it the **CA certificate** (AddOnTemplate). The SuperNode verifies the SuperLink over TLS. Cluster registration itself uses hub-signed client certificates (CSRs) |
| 17 | `FederatedLearning` defines framework, rounds, participants and storage; Waiting → Running → Completed | `FederatedLearning` defines framework, images, minimum participants and the Placement. **Waiting** until enough clusters match, **Running** once the ServerApp (hub) and the ClientApps (branches, via ManifestWorkReplicaSet) are deployed. Rounds go in the Flower run config; training starts with `flwr run`, and the run ends as `finished:completed` (`flwr ls`). The resource stays Running until deleted |
| 18 | TLS channels, options beyond NodePort | Still true. Here the SuperLink is exposed with NodePort (kind); on OpenShift, Route or LoadBalancer |
| 20 | Status goes through Waiting, Running and Completed | Waiting → Running, then the Flower run `running` → `finished:completed` |
| 20 | Flower and public data | Flower 1.34.0, Banking77 (public), Granite 4.0 350M + LoRA |
| 22 | Portuguese messages | English messages (Banking77 is English): "My transfer still hasn't arrived.", "My card was cloned.", "Why is the exchange rate on my purchase wrong?" |

## Filling in slide 22

`results/summary.md` has three tables:

1. Accuracy per queue for base / brasil-only / federated. The cards and fx_fees
   rows show the effect of federation most clearly, since those queues are rare in Brazil.
2. The three slide messages with each model's answer and confidence.
3. Candidate messages: real test messages (never seen in training) that the
   federated model routes correctly and the Brazil-only model does not.

With the default data split (`--dominance 0.9`), Brazil still sees about 55
card and 55 fx_fees messages, so the Brazil-only model gets the three short
slide messages right. The difference shows in the per-queue accuracy, and in
the candidate messages. For a starker contrast, regenerate the data with
`DOMINANCE=1.0 make data` (each branch never sees the other branches'
dominant subjects), then run again.

## Reset between takes

```bash
make reset      # deletes the FederatedLearning resource, withdraws the claims, waits for SuperNodes to go
```
