#!/usr/bin/env bash
# One-time host setup so kind nodes can receive the NVIDIA GPU (Linux + Docker).
# Run it yourself, with sudo: it changes Docker's default runtime and restarts Docker.
#
#   sudo infra/scripts/05-gpu-host.sh
#
# Requires the NVIDIA driver and the NVIDIA Container Toolkit on the host:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# Same prerequisites as NVIDIA's nvkind (https://github.com/NVIDIA/nvkind).
set -euo pipefail

[ "$(id -u)" = "0" ] || { echo "run with sudo"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: install the NVIDIA driver first"; exit 1; }
command -v nvidia-ctk >/dev/null || { echo "nvidia-ctk not found: install the NVIDIA Container Toolkit first"; exit 1; }

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Make the NVIDIA runtime Docker's default, and let a kind node request GPUs
# through a volume mount (/var/run/nvidia-container-devices/all), which is how
# extraMounts in a kind config can pass a GPU to a node.
nvidia-ctk runtime configure --runtime=docker --set-as-default --cdi.enabled
nvidia-ctk config --set accept-nvidia-visible-devices-as-volume-mounts=true --in-place
systemctl restart docker

echo "Checking: a container that asks for GPUs through the volume mount should see them"
docker run --rm -v /dev/null:/var/run/nvidia-container-devices/all ubuntu:24.04 nvidia-smi -L
