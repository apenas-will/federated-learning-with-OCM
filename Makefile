# Federated fine-tuning demo: Open Cluster Management + Flower + Granite 4.0 350M (LoRA)
# Run `make help` for the list of targets.

SHELL := /bin/bash
export PATH := $(CURDIR)/bin:$(PATH)
export FLWR_HOME := $(CURDIR)/.flwr
export FLWR_DISABLE_UPDATE_CHECK := 1
# CONTAINER_TOOL=docker|podman and GPU=auto|1|0; defaults in infra/scripts/lib.sh
PY := $(CURDIR)/.venv/bin/python
FLWR := $(CURDIR)/.venv/bin/flwr
HUB := kind-hub
S := infra/scripts

ROUNDS ?=
THREADS ?=
RUN_CONFIG := $(strip $(if $(ROUNDS),num-server-rounds=$(ROUNDS)) $(if $(THREADS),num-threads=$(THREADS)))
RUN_CONFIG_FLAG := $(if $(RUN_CONFIG),--run-config '$(RUN_CONFIG)',)

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2} /^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)}' $(MAKEFILE_LIST)

##@ One command
.PHONY: run-all simulate
run-all: ## Setup (if needed) + full OCM federated run + evaluation; everything lands in results/
	demo/run-all.sh
simulate: ## Same Flower app as local processes (no clusters), on the GPU if present; then evaluate
	$(S)/local-federation.sh '$(RUN_CONFIG)'
	@[ -f models/adapters/brasil-only/adapter_model.safetensors ] || $(PY) eval/train_local.py --branch brasil
	$(PY) eval/evaluate.py

##@ One-time setup (about 20-30 minutes, mostly downloads)
.PHONY: gpu-host
gpu-host: ## Linux + NVIDIA only, once, as root: let kind nodes use the GPU (sudo make gpu-host)
	$(S)/05-gpu-host.sh

.PHONY: setup prereqs data clusters image ocm addon controller branch-data
setup: prereqs clusters image ocm addon controller branch-data ## Everything up to (not including) the demo
prereqs: ## Check tools, fetch clusteradm and pinned upstream sources, create .venv
	$(S)/00-check-prereqs.sh
data: ## Re-create data/ from Banking77 (DOMINANCE=0.9 by default; only needed to change the split)
	$(PY) prepare_banking77.py --out data --dominance $(or $(DOMINANCE),0.9) --max-per-queue 1200
clusters: ## Local registry + kind clusters: hub, brasil, mexico, portugal, chile
	$(S)/10-clusters.sh
image: ## Base model (fp32), app image, push and pre-pull into the nodes
	$(S)/20-image.sh
ocm: ## OCM hub + klusterlets (clusteradm)
	$(S)/30-ocm.sh
addon: ## flower-addon: SuperLink with TLS + Placement-driven SuperNodes
	$(S)/40-flower-addon.sh
controller: ## federated-learning-controller
	$(S)/50-fl-controller.sh
branch-data: ## Copy each branch's data to its own cluster and serve it locally
	$(S)/60-branch-data.sh

##@ Demo steps (see docs/demo-runbook.md, or run them all with `make demo`)
.PHONY: demo reset publish-data withdraw-data fl train runs fetch-model fl-delete
demo: ## Guided, step-by-step demo for the recording (AUTO=1 for no pauses)
	demo/demo.sh
reset: ## Back to the demo's starting point (no FL resource, no claims, no SuperNodes)
	demo/reset.sh
publish-data: ## Branches publish the customer-messages ClusterClaim
	$(S)/61-publish-claims.sh
withdraw-data: ## Branches withdraw the ClusterClaim
	$(S)/61-publish-claims.sh --remove
fl: ## Create the FederatedLearning resource on the hub
	kubectl --context $(HUB) apply -f infra/manifests/hub/federatedlearning.yaml
train: ## Submit the Flower app to the SuperLink (ROUNDS=n, THREADS=n to override)
	$(FLWR) run app ocm --stream $(RUN_CONFIG_FLAG)
runs: ## List Flower runs and their status
	$(FLWR) ls ocm
fetch-model: ## Copy the federated adapter out of the ServerApp pod
	$(S)/70-fetch-model.sh
fl-delete: ## Delete the FederatedLearning resource (ServerApp + ClientApps go away)
	kubectl --context $(HUB) delete -f infra/manifests/hub/federatedlearning.yaml --ignore-not-found

##@ Evaluation (GPU if available)
.PHONY: baseline evaluate notebook
baseline: ## Train the "Brazil only" model with the same compute as one branch
	$(PY) eval/train_local.py --branch brasil
evaluate: ## Compare base / brasil-only / federated; writes results/summary.md
	$(PY) eval/evaluate.py
notebook: ## Open the validation notebook
	$(CURDIR)/.venv/bin/jupyter lab eval/validate_model.ipynb

##@ Observability
.PHONY: status watch logs data-access
status: ## One-shot view of clusters, placement, add-ons and the FL resource
	@demo/status.sh
watch: ## Refresh `make status` every 3 seconds
	@while true; do clear; demo/status.sh; sleep 3; done
logs: ## Follow ClientApp logs of a branch (BRANCH=brasil|mexico|portugal) or the hub ServerApp (BRANCH=hub)
	@if [ "$(BRANCH)" = "hub" ]; then \
	  kubectl --context $(HUB) -n bank-fl logs -f -l app.kubernetes.io/component=superexec-serverapp --tail=50; \
	else \
	  kubectl --context kind-$(BRANCH) -n flower-addon logs -f -l app.kubernetes.io/component=superexec-clientapp --tail=50; \
	fi
data-access: ## Access log of a branch's data service (BRANCH=...)
	kubectl --context kind-$(BRANCH) -n bank-data logs -f deploy/branch-data --tail=20

##@ Cleanup
.PHONY: down
down: ## Delete all clusters and the local registry
	$(S)/90-teardown.sh
