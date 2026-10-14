# Architecture

This demo runs the scenario from the talk: one bank, three country branches,
one model that routes customer messages to the right support queue, and no
customer message ever leaving its country. Everything is open source and runs
on a laptop: five [kind](https://kind.sigs.k8s.io/) clusters on Podman,
[Open Cluster Management](https://open-cluster-management.io) (OCM, the
upstream of Red Hat Advanced Cluster Management), the community
[federated-learning-controller and flower-addon](https://github.com/open-cluster-management-io/addon-contrib),
[Flower](https://flower.ai), and IBM Granite 4.0 350M fine-tuned with LoRA.

## Overview

```
 laptop ─ flwr run app ocm ──(TLS, Control API, NodePort 30093)──┐
                                                                 ▼
┌──────────────────────────────── hub ────────────────────────────────────┐
│ OCM cluster-manager: registration · placement · work · addon-manager    │
│ federated-learning-controller   (watches FederatedLearning resources)   │
│ flower-system/superlink         (Fleet API :30092 TLS, Control :30093)  │
│ bank-fl/…-serverapp             (SuperExec running the ServerApp: FedAvg)│
└──────────────▲───────────────────────▲───────────────────────▲──────────┘
   LoRA adapter│(5.7 MB/round, gRPC+TLS)│                       │
┌──────────────┴─────┐ ┌───────────────┴────┐ ┌────────────────┴───┐ ┌──────────────┐
│ brasil             │ │ mexico             │ │ portugal           │ │ chile        │
│ klusterlet         │ │ klusterlet         │ │ klusterlet         │ │ klusterlet   │
│ ClusterClaim: data │ │ ClusterClaim: data │ │ ClusterClaim: data │ │ (no data,    │
│ flower-addon:      │ │ flower-addon:      │ │ flower-addon:      │ │  no claim)   │
│  SuperNode + CA    │ │  SuperNode + CA    │ │  SuperNode + CA    │ │              │
│  SuperExec-Client  │ │  SuperExec-Client  │ │  SuperExec-Client  │ │ not selected │
│ bank-data:         │ │ bank-data:         │ │ bank-data:         │ │              │
│  branch-data svc   │ │  branch-data svc   │ │  branch-data svc   │ │              │
│  /data/bank (node) │ │  /data/bank (node) │ │  /data/bank (node) │ │              │
└────────────────────┘ └────────────────────┘ └────────────────────┘ └──────────────┘
```

## Components

| Component | Runs on | Source | Version |
|---|---|---|---|
| kind clusters (`hub`, `brasil`, `mexico`, `portugal`, `chile`) | Podman machine | kind | v0.33 |
| OCM hub (`cluster-manager`) and klusterlets | hub / branches | `clusteradm` | bundle 1.4.0 |
| federated-learning-controller | hub, `open-cluster-management` | addon-contrib @ `6199d61`, image `:latest` pinned by digest | Flower path with SuperLink/SuperNode |
| flower-addon (Helm chart, unmodified) | SuperLink on hub, SuperNodes on branches | addon-contrib @ `6199d61` | Flower images 1.34.0 |
| App image `bank-queue-router` | hub (ServerApp) and branches (ClientApp) | `app/Containerfile` | `flwr/superexec:1.34.0` + PyTorch (CUDA 12.6/13.0 with GPU, CPU otherwise) + Granite fp32 |
| NVIDIA Container Toolkit in the branch nodes (GPU mode) | brasil, mexico, portugal | `infra/scripts/10-clusters.sh`, as in NVIDIA's nvkind | from NVIDIA's apt repository |
| Flower app (ServerApp/ClientApp code) | shipped by `flwr run` as a FAB | `app/` | — |
| `branch-data` service | each data branch, `bank-data` | `infra/manifests/branch/` | python:3.13-alpine |

**Why Flower 1.34.0.** The controller and the flower-addon chart pass Flower CLI
flags that newer releases removed: `--clientappio-api-address` on the SuperNode
(gone in 1.35) and `--appio-api-address` / `--plugin-type` on the SuperExec
(gone in 1.37). 1.34.0 is the newest release that works with both, unmodified.
Upstream tests with 1.26.1.

## Lifecycle

| Step | What happens | Where to look |
|---|---|---|
| 0. Prerequisites | OCM installed, flower-addon deployed (SuperLink on the hub, add-on registered), controller running, app image in the registry | `make setup` |
| 1. Request | A `FederatedLearning` resource is created on the hub (`infra/manifests/hub/federatedlearning.yaml`) | `kubectl get federatedlearning -n bank-fl` |
| 2. Waiting | The controller creates a Placement from `spec.client.placement` and deploys the ServerApp next to the SuperLink. Phase `Waiting`: "Expected 3 clusters, but only 0 meet the criteria" | `make status` |
| 3. Selection | Branches that hold data publish a ClusterClaim. The add-on Placement selects them and the flower-addon installs a SuperNode there (with the CA certificate). Once the add-on is available, the FL Placement selects them too | `kubectl get placementdecisions -A` |
| 4. Running | The controller distributes the ClientApp to the selected branches with a ManifestWorkReplicaSet. Phase `Running`: "Assigned 3 clusters for client execution in model training" | `kubectl get manifestworkreplicaset -n bank-fl` |
| 5. Training | `flwr run app ocm` sends the app (FAB) to the SuperLink. Each round: the hub sends the global adapter, every branch trains on its own data and returns its adapter, the hub averages them (FedAvg) and asks every branch to evaluate the result on its own validation set | `flwr ls ocm`, `make logs BRANCH=…` |
| 6. Done | The run is `finished:completed`; the final adapter and the metrics history are in the ServerApp pod (`/app/output/latest`) | `make fetch-model` |
| 7. Validation | The adapter is compared with the base model and a Brazil-only model in a notebook | `make evaluate`, `eval/validate_model.ipynb` |

The resource stays `Running` until it is deleted; several `flwr run` can be
submitted against the same deployment. Deleting it removes the ServerApp, the
ManifestWorkReplicaSet and therefore the ClientApps.

## Selecting the participants (two Placements)

Both use the same idea: a branch is selected only if it **holds the data** and
**has the resources**.

1. `open-cluster-management/branches-with-customer-data`
   (`infra/manifests/hub/flower-addon-placement.yaml`) drives the flower-addon
   install strategy: which clusters get a Flower SuperNode.
2. The Placement inside the `FederatedLearning` resource selects where the
   ClientApp runs: clusters where the add-on is available **and** the data claim exists.

Data locality comes from a ClusterClaim (`customer-messages.bank.demo`) that
each branch creates **on its own cluster**; the klusterlet reports it to the
hub. The claim carries a dataset id (`banking77-brasil-v1`), never data.
`chile` has CPU and memory but no claim, so it never gets a SuperNode or a
ClientApp, even though its node has the SuperNode image pre-pulled.

## What crosses cluster boundaries

| What | From → to | Channel |
|---|---|---|
| Customer messages | nowhere: they stay on the branch node and are read through a ClusterIP service restricted by a NetworkPolicy | — |
| Global LoRA adapter (5.7 MB) | hub → branch, every round | Flower Fleet API, gRPC over TLS |
| Updated LoRA adapter (5.7 MB) + metrics (loss, accuracy, number of examples) | branch → hub, every round | same |
| App code (Flower App Bundle) | laptop → SuperLink → SuperExecs | Control API (TLS), then Fleet API |
| ClusterClaims, add-on and cluster status | branch → hub | klusterlet, mTLS with a hub-signed client certificate |
| ManifestWorks (SuperNode, ClientApp) | hub → branch | OCM work agent pulls from the hub |

The base model (Granite 4.0 350M, 1.3 GB in fp32) is part of the image every
branch already runs, so it never travels during training.

## Security

- **Cluster registration.** Each klusterlet authenticates to the hub with a
  client certificate issued through a CertificateSigningRequest that the hub
  approves (`clusteradm accept`); `kubectl get csr` shows them.
- **Flower channel.** `generate-certs.sh` from the flower-addon creates a CA and
  a SuperLink server certificate. The SuperLink serves the Fleet API (SuperNodes)
  and the Control API (`flwr` CLI) over TLS. The add-on's AddOnTemplate ships the
  CA certificate to every selected branch as a Secret, and each SuperNode
  verifies the SuperLink with it (`--root-certificates`).
- **In-cluster hops.** ServerApp ↔ SuperLink (ServerAppIo) and ClientApp ↔
  SuperNode (ClientAppIo) stay inside one cluster and are plaintext; the
  controller deploys the SuperExecs with `--insecure`.
- **Data service.** `branch-data` is ClusterIP only, read-only, runs as nobody,
  and its NetworkPolicy admits only pods from `flower-addon`.
- **Not covered here.** SuperNode authentication (Flower supports
  `--enable-supernode-auth`), per-client certificates, secure aggregation and
  differential privacy. Adapters can still leak information about the training
  data; these are the next layers for a real deployment.

## Model and training

| | |
|---|---|
| Base model | `ibm-granite/granite-4.0-350m` (Apache-2.0), 352M parameters, frozen |
| Adapter | LoRA r=8, alpha=16, dropout 0.05 on `q_proj k_proj v_proj o_proj` of all 28 layers: 1.49M parameters (0.42%), 5.7 MB |
| Prompt | `Route the bank customer message to one support queue: cards, transfers, top_ups, fx_fees, account_identity.\nMessage: {text}\nQueue:` |
| Loss | cross-entropy on the answer tokens only (queue name + end of text); only those logits are computed |
| Prediction | constrained to the five queues: the first token of each queue name is distinct, so the answer is the queue whose first token is most likely after `Queue:` |
| Per round, per branch | 25 AdamW steps, batch 4, lr 5e-4 (`[tool.flwr.app.config]` in `app/pyproject.toml`) |
| Aggregation | FedAvg over the LoRA A and B matrices, weighted by each branch's number of training examples |
| Evaluation during training | each branch scores the new global adapter on its own validation set; the hub only receives the numbers (per branch and weighted, saved in `history.json`) |
| Device | `device = "auto"`: CUDA when the process sees a GPU (baseline, evaluation, `make simulate` on a GPU server), CPU in the kind clusters |

The hub has no customer data, so there is no central evaluation during
training. The final comparison (`eval/evaluate.py`) runs on the laptop with
the public Banking77 test set.

**GPU.** With a GPU (Linux + Docker), the three data branches share the
host's NVIDIA card. Each branch node gets it through the NVIDIA runtime
(`/var/run/nvidia-container-devices/all` mount), runs the NVIDIA Container
Toolkit as containerd's default runtime, and the app image sets
`NVIDIA_VISIBLE_DEVICES=all`. The controller's ClientApp pods therefore see the GPU
without GPU requests or a RuntimeClass, which the controller does not set. With
one shared card there is no scheduling or isolation between branches; on
OpenShift the NVIDIA GPU Operator, device plugin and requests take care of that.

**Why fp32 and these settings on CPU.** Containers on a Mac cannot use the
Apple GPU. With PyTorch's oneDNN CPU kernels on Linux/arm64, a LoRA step took
20–60 s and peaked at about 9 GB per process. With oneDNN disabled it takes
about 1.5–3 s and about 2.5 GB at batch 4. Storing the base model in fp32 lets
it load by memory-mapping instead of converting bf16 at load time, which avoided
a further ~1 GB peak per branch.

## Data

`prepare_banking77.py` maps Banking77's 77 intents to five queues and splits
the training set non-IID: each branch keeps 90% of "its" queue (Brasil:
transfers, México: cards, Portugal: fx_fees), the other two branches share
the remaining 10%, and the neutral queues (top_ups, account_identity) are split
evenly. Each branch has 1,777 training and 197 validation messages; see
`data/stats.json`.

## Running without clusters

`make simulate` (`infra/scripts/local-federation.sh`) starts the same Flower
components as local processes: a SuperLink, one SuperNode per branch (with
`cluster-name` and `data-endpoint` in its node config, like the flower-addon
sets) and one localhost data server per branch. `flwr run app local` then runs
the unchanged app. It skips OCM entirely, so it shows the learning side but
not the platform side; use it to get numbers quickly on a GPU or to iterate on the app.
