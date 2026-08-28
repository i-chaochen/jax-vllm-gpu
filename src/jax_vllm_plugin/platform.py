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

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig

logger = logging.getLogger(__name__)

PLATFORM_QUALNAME = "jax_vllm_plugin.platform.JaxPlatform"

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
    device_name: str = "rocm-jax"

    # "cpu" because torch here is a CPU-only build, so torch.device(device_type)
    # must remain constructible; a ROCm torch is what we are avoiding, and an
    # invented name would raise. This does not put vLLM on its CPU code paths:
    # is_cpu() tests _enum, which is OOT.
    device_type: str = "cpu"

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
        parallel_config = vllm_config.parallel_config
        parallel_config.worker_cls = "jax_vllm_plugin.worker.JaxWorker"

        # One vLLM worker drives every GPU through a JAX mesh, so vLLM must not
        # fan out into a process per device. tensor_parallel_size is consumed as
        # a mesh axis by the runner rather than as a process count.
        if parallel_config.pipeline_parallel_size == 1:
            parallel_config.distributed_executor_backend = "uni"

        cache_config = vllm_config.cache_config
        if cache_config is not None and not cache_config.user_specified_block_size:
            cache_config.block_size = DEFAULT_TOKENS_PER_PAGE

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
