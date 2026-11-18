# SPDX-License-Identifier: Apache-2.0
"""vLLM platform plugin for JAX on AMD GPUs.

`register` is the `vllm.platform_plugins` entry point. vLLM calls it and then
resolves its *return value* as a qualified name, so it must return a string or
None -- never the Platform class itself.
"""

from __future__ import annotations

import logging
import os
from importlib.util import find_spec
from typing import TYPE_CHECKING

from vllm.platforms.interface import Platform, PlatformEnum

from . import torch_shims

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig

logger = logging.getLogger(__name__)

PLATFORM_QUALNAME = "jax_vllm_gpu.plugin.platform.JaxPlatform"

# Set to "1" to force activation or "0" to force the plugin off. Without this,
# a probe failure is indistinguishable from the wheel not being installed,
# because vLLM swallows exceptions raised by platform plugins.
FORCE_ENV_VAR = "VLLM_JAX_PLATFORM"

# Present whenever the amdgpu kernel driver has exposed a compute device. This
# is checked instead of importing jax and calling jax.devices(), which would
# initialise a backend as a side effect of mere platform detection.
_AMD_KFD_DEVICE = "/dev/kfd"

# Matches MaxText's paged_page_size default and the page sizes the aiter paged
# kernels accept (16 or 32), so vLLM's block_size and our pool agree by default.
DEFAULT_TOKENS_PER_PAGE = 16

# MaxText's attention mode for the AMD paged path. Its `forward_serve_gpu_paged`
# takes the pool and metadata as arguments, mirroring the vLLM RPA branch, which
# is exactly the externally-owned-pool arrangement this platform needs.
PAGED_ATTENTION_MODE = "gpu_paged"


def _explicit_override() -> bool | None:
    value = os.environ.get(FORCE_ENV_VAR)
    if value is None:
        return None
    return value.strip().lower() in ("1", "true", "yes", "on")


def register() -> str | None:
    """Return this platform's qualname if we should own the process, else None."""
    override = _explicit_override()
    if override is False:
        logger.debug("JAX platform disabled by %s.", FORCE_ENV_VAR)
        return None
    if override is True:
        logger.info("JAX platform forced on by %s.", FORCE_ENV_VAR)
        return PLATFORM_QUALNAME

    if find_spec("jax") is None:
        logger.debug("JAX platform unavailable: jax is not importable.")
        return None

    if not os.path.exists(_AMD_KFD_DEVICE):
        logger.debug(
            "JAX platform unavailable: %s is absent, so no AMD GPU is visible.",
            _AMD_KFD_DEVICE,
        )
        return None

    logger.debug("JAX platform is available.")
    return PLATFORM_QUALNAME


class JaxPlatform(Platform):
    """AMD GPU platform whose compute is JAX rather than PyTorch.

    JAX owns the accelerators. torch is present only because vLLM requires it,
    and is deliberately a CPU-only build: a ROCm torch would give the process a
    second HIP runtime alongside JAX's, which aborts during RCCL clique setup.
    """

    _enum = PlatformEnum.OOT

    # Both of these are "cpu" because torch here is a CPU-only build, so both
    # `torch.device(device_type)` and -- for an out-of-tree platform --
    # `torch.device(f"{device_name}:{index}")` in `GroupCoordinator` must stay
    # constructible. A descriptive name like "rocm-jax" reads better in logs and
    # raises `Invalid device string` the moment a process group is created.
    #
    # This does not put vLLM on its CPU code paths: `is_cpu()` tests `_enum`,
    # which is OOT. Nor does it describe where the model runs -- JAX owns the
    # GPUs and torch is never told about them.
    device_name: str = "cpu"
    device_type: str = "cpu"

    # For humans, since device_name can no longer say it.
    display_name: str = "rocm-jax"

    # XLA, as for the TPU platform: both defer compilation to JAX.
    dispatch_key: str = "XLA"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "HIP_VISIBLE_DEVICES"

    # gloo, not nccl. The torch process group exists only to satisfy vLLM's
    # parallel_state at world size 1; every real collective is JAX's.
    dist_backend: str = "gloo"

    # torch.compile has nothing to compile on this platform.
    simple_compile_backend: str = "eager"

    supported_quantization: list[str] = []

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Called very early in vLLM startup; this is where worker_cls is set."""
        # Before anything else, because it makes teardown survivable and
        # teardown runs even for a startup that fails after this point.
        torch_shims.install()

        parallel_config = vllm_config.parallel_config
        parallel_config.worker_cls = "jax_vllm_gpu.plugin.worker.JaxWorker"

        # One vLLM worker drives every GPU through a JAX mesh, so vLLM must not
        # fan out into a process per device. tensor_parallel_size is consumed as
        # a mesh axis by the runner rather than as a process count.
        if parallel_config.pipeline_parallel_size == 1:
            parallel_config.distributed_executor_backend = "uni"

        cache_config = vllm_config.cache_config
        if cache_config is not None and not cache_config.user_specified_block_size:
            cache_config.block_size = DEFAULT_TOKENS_PER_PAGE

        cls._select_paged_attention(vllm_config)

    @staticmethod
    def _select_paged_attention(vllm_config: "VllmConfig") -> None:
        """Point MaxText at the GPU paged attention path.

        MaxText's `inference/vllm.yml` defaults to `vllm_rpa`, which is TPU
        Pallas ragged paged attention. Its adapter merges
        `additional_config["maxtext_config"]` into `pyconfig.initialize`, so the
        mode can be selected here rather than by editing a config MaxText shares
        with the TPU plugin.

        An explicit setting is left alone: a caller asking for a different
        attention deliberately should not be overridden silently.
        """
        additional = getattr(vllm_config, "additional_config", None)
        if additional is None:
            return
        block_size = vllm_config.cache_config.block_size
        maxtext_config = additional.setdefault("maxtext_config", {})
        maxtext_config.setdefault("attention", PAGED_ATTENTION_MODE)
        maxtext_config.setdefault("paged_page_size", block_size)

        if maxtext_config["attention"] != PAGED_ATTENTION_MODE:
            return

        # MaxText's own context bound. Left at its default it is shorter than
        # what vLLM will admit, and `gpu_paged` validation compares the two, so
        # a long request would be rejected by the model rather than by the
        # scheduler that is supposed to be bounding it.
        max_model_len = int(vllm_config.model_config.max_model_len)
        maxtext_config.setdefault("max_target_length", max_model_len)

        # Inert here, and required to be set. `paged_num_blocks` sizes the pool
        # MaxText would allocate for itself, but under this platform vLLM owns
        # allocation and `attentions.py` skips `init_kv_caches` entirely -- so
        # nothing is allocated from this number. It still has to clear
        # validation, which insists a single longest request fits.
        maxtext_config.setdefault(
            "paged_num_blocks", -(-max_model_len // block_size)
        )

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        **kwargs,
    ) -> str:
        raise NotImplementedError(
            "The paged attention backend is not implemented yet. It will wrap "
            "the jax-aiter paged kernels via AiterPagedAttentionAbiV1."
        )
