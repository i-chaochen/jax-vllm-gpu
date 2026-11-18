# SPDX-License-Identifier: Apache-2.0
"""The vLLM worker, named by `JaxPlatform.check_and_update_config`.

One worker drives every GPU through a JAX mesh. That is why the platform forces
`distributed_executor_backend="uni"`: vLLM must not fan out into a process per
device, because `tensor_parallel_size` is consumed here as a mesh axis instead
of as a process count.

**torch.distributed still gets initialised, at world size one, over gloo.** It
carries no traffic. vLLM's `parallel_state` is a hard dependency of a good deal
of model and sampling code that runs even in a single-process deployment, and it
refuses to answer before a process group exists. gloo is the only backend that a
CPU-only torch can offer, and a CPU-only torch is the point: a ROCm build would
put a second HIP runtime in the process alongside JAX's, which aborts during
RCCL clique setup.

**Subclassing `WorkerBase`, not the GPU `Worker`.** The latter's `init_device`
raises unless `device_type == "cuda"`, so nothing below it is reachable.
`CPUWorker` is the override pattern this follows -- except that two `WorkerBase`
defaults also have to go, because both reach for hardware this platform does not
present to torch: `synchronize_device` calls `torch.accelerator`, and
`get_supported_kv_cache_layouts` introspects GPU attention backend classes.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import torch

from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

logger = logging.getLogger(__name__)

# Our pool is [num_pages, tokens_per_page, num_kv_heads, head_dim]: a page is a
# run of tokens, each holding every head. That is vLLM's NHD.
KV_CACHE_LAYOUT = "NHD"


class JaxWorker(WorkerBase):
    """Owns the JAX mesh, the KV pool and the model runner."""

    def __init__(
        self,
        vllm_config: Any,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.mesh: Any = None
        self.model_runner: Any = None

    # ------------------------------------------------------------- lifecycle

    def init_device(self) -> None:
        import jax

        from .mesh import build_mesh
        from .model_runner import JaxModelRunner

        # torch's device, not JAX's. torch here holds no accelerator at all, and
        # `torch.device(device_type)` has to stay constructible for the code
        # paths that ask the platform where tensors live.
        self.device = torch.device(self.current_platform.device_type)

        self._init_torch_distributed()

        tensor_parallel_size = int(self.parallel_config.tensor_parallel_size)
        self.mesh = build_mesh(tensor_parallel_size, devices=jax.devices())
        self.model_runner = JaxModelRunner(self.vllm_config, self.mesh)

    def _init_torch_distributed(self) -> None:
        """A world-size-one gloo group, purely to satisfy `parallel_state`."""
        from vllm.config import set_current_vllm_config
        from vllm.distributed.parallel_state import (
            ensure_model_parallel_initialized,
            init_distributed_environment,
        )

        # vLLM's own data parallelism is not in play -- JAX owns every axis of
        # parallelism here -- but the value is read during init, so it is forced
        # to one for the duration and restored afterwards rather than mutated.
        parallel_config = self.parallel_config
        saved_dp_size = parallel_config.data_parallel_size
        parallel_config.data_parallel_size = 1

        handle, path = tempfile.mkstemp()
        os.close(handle)
        try:
            with set_current_vllm_config(self.vllm_config):
                init_distributed_environment(
                    world_size=1,
                    rank=0,
                    local_rank=0,
                    distributed_init_method=f"file://{path}",
                    backend="gloo",
                )
                ensure_model_parallel_initialized(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                )
        finally:
            parallel_config.data_parallel_size = saved_dp_size

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> CompilationTimes:
        # Nothing is precompiled yet. Each bucketed shape compiles on first use,
        # which the ladder bounds; warming them ahead of time is a latency
        # improvement rather than a correctness one, so it is deferred rather
        # than guessed at before the shape set has been measured.
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=0.0,
        )

    # ---------------------------------------------------------------- memory

    def determine_available_memory(self) -> int:
        """Bytes of device memory the KV pool may occupy, across all devices.

        Totalled rather than per-device on purpose. vLLM turns this into a block
        count using a page size computed from the *unsharded* head count, so the
        arithmetic only closes if the budget is unsharded too: the pool it sizes
        is then split across the mesh, and each device holds its own fraction.
        """
        import jax

        devices = self.mesh.devices.flatten().tolist() if self.mesh else jax.devices()
        utilisation = float(self.cache_config.gpu_memory_utilization)

        total_limit = 0
        total_used = 0
        for device in devices:
            stats = device.memory_stats() or {}
            limit = stats.get("bytes_limit")
            used = stats.get("bytes_in_use")
            if limit is None or used is None:
                raise RuntimeError(
                    f"device {device} reported no memory statistics "
                    f"({sorted(stats)}), so the KV budget cannot be computed"
                )
            total_limit += int(limit)
            total_used += int(used)

        available = int(total_limit * utilisation) - total_used
        logger.info(
            "KV budget: %.2f GiB of %.2f GiB across %d device(s) at "
            "utilisation %.2f, with %.2f GiB already held by weights.",
            available / 2**30,
            total_limit / 2**30,
            len(devices),
            utilisation,
            total_used / 2**30,
        )
        if available <= 0:
            raise ValueError(
                f"no memory left for the KV cache: {total_used / 2**30:.2f} GiB "
                f"is in use but utilisation {utilisation} of "
                f"{total_limit / 2**30:.2f} GiB allows only "
                f"{total_limit * utilisation / 2**30:.2f} GiB. Raise "
                "gpu_memory_utilization or shrink the model."
            )
        return available

    def get_cache_block_size_bytes(self) -> int:
        """One block's cost across every layer, summed over the whole mesh."""
        layout = self.model_runner.layout
        if layout is None:
            raise RuntimeError("the KV pool has not been allocated yet")
        return (
            2
            * layout.num_layers
            * layout.bytes_per_page()
            * layout.kv_head_shards
            * max(layout.pool_replicas, 1)
        )

    # -------------------------------------------------------------- kv cache

    def get_kv_cache_spec(self) -> dict[str, Any]:
        return self.model_runner.get_kv_cache_spec()

    def get_supported_kv_cache_layouts(self) -> list[str]:
        # The base implementation asks each registered attention backend what it
        # supports. There are no torch attention backends in this process -- the
        # kernels are jax-aiter, reached through MaxText -- so it would resolve
        # against an empty set and fall back to a GPU default.
        return [KV_CACHE_LAYOUT]

    def initialize_from_config(self, kv_cache_config: Any) -> None:
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.model_runner.initialize_kv_cache(kv_cache_config)

    # ----------------------------------------------------------- the request

    def execute_model(self, scheduler_output: Any) -> Any:
        """Runs the forward pass and returns None, leaving the tokens to
        `sample_tokens`. See the runner for why None is the only answer that
        works on both of the engine's step paths."""
        return self.model_runner.execute_model(scheduler_output)

    def sample_tokens(self, grammar_output: Any) -> Any:
        return self.model_runner.sample_tokens(grammar_output)

    def execute_dummy_batch(self) -> None:
        return

    def get_model(self) -> Any:
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[str, ...]:
        return ("generate",)

    # ------------------------------------------------------------ device ops

    def synchronize_device(self) -> None:
        """Wait for JAX, since torch here has no accelerator to wait on.

        The base implementation calls `torch.accelerator.synchronize()`, which
        with a CPU-only torch is a no-op -- it would let a caller believe device
        work had finished while the JAX queue was still draining.

        Enqueueing a trivial array per device and blocking on it is enough:
        dispatch to a given device is ordered, so anything already submitted has
        run by the time this lands.
        """
        import jax

        devices = self.mesh.devices.flatten().tolist() if self.mesh else jax.devices()
        for device in devices:
            jax.block_until_ready(jax.device_put(0, device))

    def check_health(self) -> None:
        return
