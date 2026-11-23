#!/usr/bin/env bash
# Starts the vLLM-XLA-GPU development container with all GPUs: TheRock 10.1, the
# image jax-titan's CI runs. Adds a user matching the host uid, so files written
# to the workspace keep the host owner. Run on the host.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
NAME=${NAME:-${USER}_therock-10.1_vllm_xla_gpu}
IMAGE=${IMAGE:-ghcr.io/rocm/jax-base-ubu24.therock-10.1@sha256:9abb85fde7485b89db61f87d5f583bb1c3b980d479fff7d6d7ed988d652c57b3}
# The repository's parent holds the sibling checkouts (vllm, jax-titan) and
# .vllm_xla_gpu. DATA holds checkpoints and build trees; both are mounted at the
# same path, so paths read the same inside and out.
WORKSPACE=${WORKSPACE:-$(dirname "${REPO}")}
DATA=${DATA:-/data}
VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)

mounts=(-v "${WORKSPACE}:${WORKSPACE}")
if [[ -d ${DATA} ]]; then
  mounts+=(-v "${DATA}:${DATA}")
fi

docker run -d --name "${NAME}" \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "${VIDEO_GID}" --group-add "${RENDER_GID}" \
  --ipc=host --shm-size=64G --security-opt seccomp=unconfined \
  --cap-add=SYS_PTRACE --network=host \
  "${mounts[@]}" -w "${REPO}" \
  "${IMAGE}" sleep infinity

docker exec "${NAME}" bash -lc "
  getent group ${VIDEO_GID} || groupadd -g ${VIDEO_GID} video
  getent group ${RENDER_GID} || groupadd -g ${RENDER_GID} render
  groupadd -g $(id -g) $(id -gn)
  useradd -u $(id -u) -g $(id -g) -G ${VIDEO_GID},${RENDER_GID} \
    -d /home/${USER} -M -s /bin/bash ${USER}
  mkdir -p /home/${USER} && chown $(id -u):$(id -g) /home/${USER}"

echo "docker exec -it -u ${USER} ${NAME} bash"
