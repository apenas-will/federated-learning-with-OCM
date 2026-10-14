# This demo vs. OpenShift + Red Hat ACM

The demo uses only open source, community components on a single machine. The
architecture is the same one you would run on OpenShift with Red Hat Advanced
Cluster Management (ACM), which is built on Open Cluster Management (OCM).
This table lists what changes and what the platform adds.

| Area | This demo | OpenShift + Red Hat ACM (+ OpenShift AI) |
|---|---|---|
| Clusters | Five kind clusters in containers on one host | One OpenShift cluster per country, on premises or in the cloud, created or imported and upgraded from ACM |
| Multicluster management | OCM 1.4 installed with `clusteradm` | ACM: supported product on top of OCM, with a web console for clusters, add-ons and Placements, plus search, governance policies and GitOps-based application lifecycle |
| Federated learning control plane | Community `federated-learning-controller` and `flower-addon` (addon-contrib), pinned commit | Same building blocks; a Flower add-on for ACM is tracked in Red Hat Jira (ACM-29351). Check its status before promising product support |
| Selecting participants | ClusterClaim for data locality + a CEL check on allocatable CPU | Same Placement API; ACM can also score clusters by real resource usage (AddOnPlacementScore) and select GPU clusters by labels or claims |
| Exposing the aggregator | SuperLink as a NodePort, mapped to the laptop by kind | Route (TLS passthrough for gRPC) or LoadBalancer in front of the SuperLink |
| Certificates | Self-signed CA from the upstream `generate-certs.sh`, distributed to branches by the add-on | Certificates from the cert-manager Operator or the cluster's PKI; the same add-on mechanism distributes the CA. Cluster registration uses hub-signed client certificates in both cases |
| Observability | `make status`, `kubectl logs`, Flower logs | ACM observability (multicluster metrics and dashboards); the FL controller also ships an observability sidecar and an OpenTelemetry add-on |
| Images and registry | Local registry container, images built with Podman | Quay or the OpenShift internal registry, image signing and scanning |
| Pod security | kind defaults | Restricted SecurityContextConstraints: images must run with an arbitrary UID |
| Branch data | JSONL on the node, served by a small read-only HTTP service, NetworkPolicy | OpenShift Data Foundation, object storage or databases with RBAC, NetworkPolicies and audit |
| Trained model | Copied out of the ServerApp pod with `kubectl cp` | Persistent storage or S3 for the artifact, registered in the OpenShift AI model registry |
| Validation | Jupyter notebook on the laptop | OpenShift AI workbenches next to the data and the model registry |
| Serving | Not covered | OpenShift AI model serving (KServe, vLLM), which can serve the base model with LoRA adapters |
| GPUs | None inside the clusters (CPU training) | NVIDIA GPU Operator on the branch clusters; Placement can require GPUs |
| Support | Community | Red Hat support for OpenShift, ACM and OpenShift AI |

## What does not change

- The hub-and-spoke, pull-based model: the hub never needs inbound access to
  the branches, which matches how the Flower SuperNodes connect to the SuperLink.
- The `FederatedLearning` resource and the Placement that decides who participates.
- The Flower app: the same ServerApp and ClientApp, shipped with `flwr run`.
- What crosses borders: LoRA adapters and metrics, never customer messages.

## Gaps to close for a real deployment

These apply on any platform:

- SuperNode authentication, so only known branches can join (Flower supports it; the chart does not enable it).
- Secure aggregation or differential privacy, since adapters can still leak information about training data.
- An approval step before a run starts, an audit trail of runs, and a model card for the result.
