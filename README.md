# Federated fine-tuning across clusters: Open Cluster Management + Flower

Demo for the talk *Aprendizado federado com gerenciamento multicluster*
(based on the Red Hat blog post
[Empowering federated learning with multicluster management](https://www.redhat.com/en/blog/empowering-federated-learning-multicluster-management)).

An international bank has branches in **Brasil**, **México** and **Portugal**,
each with its own Kubernetes cluster and its own customer messages. Each
branch's traffic is dominated by a different subject: transfers in Brasil,
cards in México, exchange rates and fees in Portugal. The bank wants **one
model** that routes every message to the right support queue, and **no
message may leave its country**. A fourth cluster, **Chile**, has compute but
no data, and must stay out of training.

The demo fine-tunes **IBM Granite 4.0 350M** with **LoRA** using **Flower**,
orchestrated by **Open Cluster Management** (the upstream of Red Hat Advanced
Cluster Management) and the community **federated-learning-controller** and
**flower-addon**. Everything is open source and runs in local
[kind](https://kind.sigs.k8s.io/) clusters. See
[docs/openshift-differences.md](docs/openshift-differences.md) for what
OpenShift and Red Hat ACM add on top.

```
            hub: OCM + FL controller + Flower SuperLink (aggregator)
                 ▲ 5.7 MB LoRA adapter per branch per round (gRPC + TLS)
     ┌───────────┼──────────────┬───────────────┐
  brasil       mexico        portugal         chile
  data ✓       data ✓        data ✓           no data → not selected
```

Details: [docs/architecture.md](docs/architecture.md).

## Three ways to run it

| Command | What it does | Time (first run) |
|---|---|---|
| `make run-all` | Builds everything that is missing, runs the whole flow through OCM without prompts (Waiting → claims → Running → `flwr run` → fetch → evaluate) and collects metrics and evidence in `results/` | 30–60 min the first time (mostly downloads), much less afterwards |
| `make demo` | The same flow step by step, pausing before each command, for the recording. Run `make setup` and `make baseline` beforehand, and `make reset` between takes | ~10 min |
| `make simulate` | The same Flower app as local processes (SuperLink + 3 SuperNodes, no Kubernetes), on the GPU if there is one; then evaluates. Quick way to get the numbers | a few minutes on a GPU |

Results end up in `results/summary.md` (tables ready for the slides).

## Requirements

- Linux (x86_64 or arm64) or macOS, with **Docker** or **rootful Podman**
  (`CONTAINER_TOOL=docker|podman`; default: Docker when there is an NVIDIA GPU,
  otherwise Podman if installed).
- At least 8 CPUs and ~12 GB of RAM available to containers, ~60 GB of disk
  (the CUDA image is large and is pulled into four nodes).
- `kind`, `kubectl`, `helm`, `uv`, `git`, `openssl`, `curl`. `clusteradm`
  (v1.4.0) and the pinned upstream sources are downloaded by `make prereqs`.
- Internet access on the first run (images, the Granite model, Python packages).
- Optional but recommended: an NVIDIA GPU on Linux with Docker. The three
  branch clusters then train on it, and so do the baseline, the evaluation and
  `make simulate` (see [Using the GPU](#using-the-gpu)). Without one,
  everything runs on CPU, only slower.

### Running on a Linux server with an NVIDIA GPU

Target machine for this demo: Core i7, 32 GB RAM, RTX A4000 (16 GB), Linux, Docker Engine.

```bash
# 1. NVIDIA driver, Docker Engine and the NVIDIA Container Toolkit, from your
#    distribution / NVIDIA docs:
#    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
#    Add your user to the docker group (sudo usermod -aG docker $USER, then log in again).
nvidia-smi                                   # driver OK?

# 2. CLI tools:
curl -Lo /usr/local/bin/kind https://kind.sigs.k8s.io/dl/v0.33.0/kind-linux-amd64 && chmod +x /usr/local/bin/kind
curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && sudo install kubectl /usr/local/bin/
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Five kind clusters need more inotify watches than most distributions allow:
sudo sysctl fs.inotify.max_user_watches=524288 fs.inotify.max_user_instances=512

# 4. Once: let kind nodes receive the GPU (makes nvidia Docker's default runtime, restarts Docker)
git clone <this repo> && cd fed_learning_ocm
sudo make gpu-host

# 5. Run and collect the metrics
make run-all
cat results/summary.md
```

Without a GPU, skip step 4 (or force CPU with `GPU=0 make run-all`).

Use the arm64 binaries on an arm64 server. If you run `flwr` or `kubectl`
from another machine, tunnel port 30093 (Flower Control API) and the kind API
servers, or simply run everything on the server.

### Reference run: exact commands (Ubuntu 24.04, RTX 4000 Ada)

These are the exact commands used for a verified run on Ubuntu 24.04.4 LTS
(x86_64, 28 CPUs, 31 GiB RAM), NVIDIA RTX 4000 Ada Generation (20 GB),
driver 580.178.04 (CUDA 13.0), starting from a machine with no Docker, kind,
kubectl, helm or uv.

CLI tools, installed into `~/.local/bin` (no sudo needed):

```bash
mkdir -p ~/.local/bin && cd ~/.local/bin
curl -sSLo kind https://kind.sigs.k8s.io/dl/v0.33.0/kind-linux-amd64 && chmod +x kind
curl -sSLo kubectl "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && chmod +x kubectl
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | HELM_INSTALL_DIR=$HOME/.local/bin USE_SUDO=false bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Result: kind v0.33.0, kubectl v1.37.1, helm v3.22.0, uv 0.13.1
```

Docker Engine, NVIDIA Container Toolkit, inotify limits and GPU host setup (sudo):

```bash
# Docker Engine (official repository)
sudo apt-get update && sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu noble stable" | sudo tee /etc/apt/sources.list.d/docker.list
# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin nvidia-container-toolkit
sudo usermod -aG docker $USER
# inotify limits, persistent across reboots
printf 'fs.inotify.max_user_watches=524288\nfs.inotify.max_user_instances=512\n' | sudo tee /etc/sysctl.d/99-kind.conf && sudo sysctl --system
# Let kind nodes receive the GPU (prints "GPU 0: NVIDIA RTX 4000 Ada ...")
cd ~/Documents/federated-learning-with-OCM && sudo make gpu-host
```

Check (result: Docker 29.9.1, default runtime `nvidia`):

```bash
sg docker -c 'docker info --format "{{.ServerVersion}} {{.DefaultRuntime}}"'
grep accept-nvidia /etc/nvidia-container-runtime/config.toml
sysctl fs.inotify.max_user_watches fs.inotify.max_user_instances
```

Full run. `sg docker` picks up the new `docker` group without logging out
again; after a new login, plain `make run-all` is the same:

```bash
cd ~/Documents/federated-learning-with-OCM
sg docker -c 'make run-all' > run-all.log 2>&1   # follow with: tail -f run-all.log
cat results/summary.md
```

The first attempts on this machine stopped on three bugs, fixed in the scripts
since then: Helm refusing the `flower-system` namespace created by the
upstream certificate script (`40-flower-addon.sh`), `run-all.sh` waiting on a
label that only the ClientApp pods carry, and the fp32 model being written as
`0600` and unreadable by the non-root SuperExec user (`20-image.sh`). With the
fixes, `make run-all` completed (exit code 0). From the first `make run-all`
to `results/summary.md`, including the three failed attempts, took about
66 minutes, mostly first-time downloads and the CUDA image build.

`results/timings.tsv` of the successful run. Setup took seconds because the
earlier attempts had already done it:

| Phase | Seconds |
|---|---|
| setup (00..60, already done) | 39 |
| reset + FederatedLearning → Waiting | 13 |
| publish data claims → Running | 72 |
| federated training (`flwr run`, 5 rounds) | 122 |
| brasil-only baseline | 11 |
| evaluation | 32 |

Placement selected `brasil,mexico,portugal`; chile was left out. The GPU test
in `20-image.sh` reported `cuda True NVIDIA RTX 4000 Ada Generation` in all
three branches, and each branch sent a 5.69 MB LoRA update per round.
Accuracy on the balanced test set (1800 messages), from `results/summary.md`:

| Queue | base | brasil-only | federated |
|---|---|---|---|
| cards | 100% | 75% | 63% |
| transfers | 0% | 97% | 82% |
| top_ups | 0% | 81% | 91% |
| fx_fees | 0% | 23% | 88% |
| account_identity | 1% | 86% | 97% |
| **overall** | **20%** | **72%** | **84%** |

### On a Mac

```bash
brew install podman kind kubernetes-cli helm uv
podman machine init --cpus 8 --memory 12288 --disk-size 120 --rootful && podman machine start
make run-all
```

A 16 GB Mac can run the clusters, but training is slow and the machine is
busy for the whole run; a server is the better place.

## What `make run-all` collects

| File | Content |
|---|---|
| `results/summary.md` | Accuracy per queue for base / brasil-only / federated, the slide messages, and real test messages the federated model routes correctly and the Brazil-only model does not |
| `results/evaluation.json` | All numbers, including per-message predictions |
| `results/history-federated.json` | Per round: each branch's training loss, update size and validation accuracy, and the hub's weighted average |
| `results/timings.tsv` | Duration of every phase |
| `results/cluster/` | Status in Waiting and in Running, Placement decisions (chile excluded), add-ons, CSRs, the SuperNode TLS arguments |
| `results/logs/` | `flwr run` output, ServerApp and ClientApp logs, and each branch's data-service access log (shows only local reads) |

Knobs: `GPU=0|1` (default auto), `THREADS=8 make run-all` (CPU threads per branch, CPU mode), `ROUNDS=8 make train`,
`DOMINANCE=1.0 make data` (how strongly each branch is dominated by its subject;
regenerate before running), local steps, batch size and learning rate in
`[tool.flwr.app.config]` of `app/pyproject.toml`.

## Using the GPU

With `GPU=auto` (default), when `nvidia-smi` works and the runtime is Docker,
the three data branches train on the host GPU. They share the same physical
card (each ClientApp needs roughly 3 GB of the A4000's 16 GB); the hub and
chile get no GPU. It works the way NVIDIA's [nvkind](https://github.com/NVIDIA/nvkind) does:

1. `make gpu-host` (once, as root) makes `nvidia` Docker's default runtime and
   lets a container request GPUs through a volume mount.
2. `make clusters` creates the brasil, mexico and portugal nodes with the
   mount `/dev/null:/var/run/nvidia-container-devices/all`, installs the NVIDIA
   Container Toolkit inside each node and makes `nvidia` containerd's default runtime.
3. `make image` builds the app image with CUDA PyTorch (12.6 or 13.0, matching
   the driver) and `NVIDIA_VISIBLE_DEVICES=all`. That is what lets the ClientApp
   pods, which the controller creates without GPU requests or a RuntimeClass, see the GPU.
   A test pod in each branch confirms `torch.cuda.is_available()`.

If a branch does not see the GPU, `make image` warns and training there falls
back to CPU; nothing else changes. The code picks the device itself
(`device = "auto"` in the run config), so `make baseline`, `make evaluate`,
the notebook and `make simulate` use the GPU as well.

## Repository layout

```
app/                     Flower app (ServerApp, ClientApp, LoRA training) + SuperExec image
  bank_router/task.py    model, data, training and constrained prediction
  pyproject.toml         Flower app config: rounds, local steps, batch size, lr, ...
data/                    Banking77 split into branches (prepare_banking77.py)
eval/                    Brazil-only baseline, evaluation, validation notebook
infra/scripts/           setup steps 00..70, local-federation.sh, teardown
infra/manifests/hub/     FederatedLearning resource, add-on Placement, bindings
infra/manifests/branch/  branch-local data service + NetworkPolicy
demo/                    run-all.sh, demo.sh (guided), reset.sh, status.sh
docs/                    architecture, demo runbook, OpenShift differences, troubleshooting
```

## Pinned versions

| Component | Version |
|---|---|
| OCM (clusteradm + bundle) | 1.4.0 |
| addon-contrib (FL controller, flower-addon) | `6199d61` (2026-09-28); controller image `:latest` pinned by digest |
| Flower | 1.34.0 (newest release compatible with the controller and chart flags) |
| kind | v0.33 |
| PyTorch / Transformers / PEFT | 2.14.1 / 5.19.0 / 0.21.2 |
| Base model | `ibm-granite/granite-4.0-350m` (Apache-2.0) |
| Data | [Banking77](https://huggingface.co/datasets/PolyAI/banking77) (public, CC-BY-4.0) |

## Docs

- [docs/architecture.md](docs/architecture.md): components, lifecycle, what crosses cluster boundaries, security, model
- [docs/demo-runbook.md](docs/demo-runbook.md): recording the demo, step by step, with the matching slides
- [docs/openshift-differences.md](docs/openshift-differences.md): this open source setup vs. OpenShift + Red Hat ACM
- [docs/troubleshooting.md](docs/troubleshooting.md)
