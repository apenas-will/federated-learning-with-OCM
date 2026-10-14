# Troubleshooting

## Where things stand

Validated on an Apple M4 (16 GB, Podman):

- the ML code, the Flower app and the run-completion handling, through a local
  Flower deployment (`make simulate`: SuperLink + 3 SuperNodes + data servers);
- the container image build, the local registry, image pulls from the kind
  nodes, creation of the five kind clusters, and `clusteradm init` on the hub;
- the GPU/runtime detection logic (with a simulated `nvidia-smi`).

Not yet run end to end: the klusterlet join (on the 16 GB Mac the VM ran out of
CPU and the join timed out), the flower-addon with TLS, the
federated-learning-controller, the Placements, the data service, `flwr run`
against the hub, and the whole GPU path (no NVIDIA hardware was available).
The first `make run-all` on the server is the first full run, so check each
phase's output; `results/timings.tsv` shows where it stopped.

## Setup

**`clusteradm join` times out / registration agent restarts.** The machine is
short on CPU. The script keeps waiting for the registration request; if it
still fails, re-run `make ocm` (it skips clusters that already registered).
Check with `kubectl --context kind-brasil -n open-cluster-management-agent get pods`.

**kind clusters fail to start on Linux ("too many open files").** Raise the
inotify limits: `sudo sysctl fs.inotify.max_user_watches=524288 fs.inotify.max_user_instances=512`.

**Podman on Linux.** kind needs rootful Podman: run with `sudo -E make …`, or
use `CONTAINER_TOOL=docker`.

**Images not found in the nodes (`ErrImagePull` for `localhost:5001/...`).**
The controller deploys the SuperExecs with `imagePullPolicy: Always`, so the
nodes must reach the local registry. Re-run `make clusters` (it reconnects the
registry to the `kind` network and rewrites each node's registry config), then `make image`.

**The CEL resource check is skipped.** `40-flower-addon.sh` tests the CEL
expression on a throwaway Placement and falls back to the data claim alone if
this OCM version cannot evaluate it. The demo still works; only the
"has resources" criterion is gone.

## GPU

**`Docker is not set up to hand the GPU to kind nodes`.** Run `sudo make gpu-host`
once. It needs the NVIDIA driver (`nvidia-smi`) and the NVIDIA Container Toolkit
(`nvidia-ctk`) on the host. To train on CPU instead: `GPU=0 make run-all`.

**`<node> does not see the GPU`** while creating clusters. The node was created
before `make gpu-host`, or Docker was not restarted. Delete and recreate:
`make down && make run-all`.

**`pods do not see the GPU` warning after `make image`.** Inside the node, check
`docker exec brasil-control-plane nvidia-smi` (the node sees it?) and
`docker exec brasil-control-plane grep -n default_runtime_name /etc/containerd/config.toml`
(should be `nvidia`). Training still works there, on CPU.

**CUDA available on the host but not in the pods.** The image picks CUDA 13.0
when the driver supports it, otherwise 12.6. Rebuild with an explicit index if
needed: `TORCH_INDEX=https://download.pytorch.org/whl/cu126 make image`.

**GPU out of memory.** The three ClientApps share one card. Lower `batch-size`
in `app/pyproject.toml`, or check that nothing else is using it (`nvidia-smi`).

## During the demo

**FederatedLearning stays in Waiting.**

1. `make status`: do brasil/mexico/portugal show the claim and `available`
   for the flower-addon?
2. No claim: `make publish-data`.
3. Claim but no add-on: `kubectl --context kind-hub get placementdecisions -n open-cluster-management`
   and `kubectl --context kind-brasil -n flower-addon get pods` (the SuperNode
   must be Running and connected to the SuperLink).
4. Add-on available but still Waiting: check the controller logs:
   `kubectl --context kind-hub -n open-cluster-management logs deploy/federated-learning-controller`.

**`flwr run` cannot connect.** The hub maps NodePort 30093 to
`127.0.0.1:30093`; `.flwr/config.toml` (generated) points there and uses
`.flwr/ca.crt`. TLS errors usually mean the CA was regenerated: re-run `make addon`.

**A round never finishes.** One ClientApp is stuck or crashed:
`make logs BRANCH=<branch>`. Out of memory shows as a restarted pod
(`kubectl --context kind-<branch> -n flower-addon get pods`); lower `THREADS`
or give the container runtime more memory.

**Training is very slow on CPU.** The app disables PyTorch's oneDNN kernels
(on Linux/arm64 they made LoRA steps ~10x slower and used ~9 GB). If you change
`task.py`, keep `torch.backends.mkldnn.enabled = False`. More threads per
branch help: `THREADS=8 make train`.

**`make fetch-model` finds no run.** The adapter lives in the ServerApp pod
(no volume). If the pod restarted or the FederatedLearning resource was
deleted, the result is gone; run `make train` again.

## Clean up

```bash
make reset   # back to the demo's starting point
make down    # delete clusters and registry (keeps models/, data/, results/)
```
