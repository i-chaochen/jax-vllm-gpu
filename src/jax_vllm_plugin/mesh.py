# SPDX-License-Identifier: Apache-2.0
"""The JAX device mesh, built from vLLM's parallel config.

There is exactly one vLLM worker and it drives every GPU through this mesh, so
`tensor_parallel_size` is consumed as a mesh axis rather than as a process
count. The platform already forces `distributed_executor_backend="uni"` to make
that true.

**Axis names are not ours to choose.** MaxText loads
`configs/inference/vllm.yml`, whose `logical_axis_rules` name the mesh axes
`data`, `attn_dp`, `model`, `expert` and `attn_dp_expert`. Those rules are
applied by the model itself, so a mesh spelling the tensor-parallel axis
anything else -- `tensor`, say, which is what MaxText's *own* configs use --
resolves to no mesh axis and silently replicates what should be sharded.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)

# In `configs/inference/vllm.yml` order. The mesh must carry all five even when
# four of them are size 1, because the logical rules reference them by name.
MESH_AXES = ("data", "attn_dp", "model", "expert", "attn_dp_expert")

# The axis vLLM's tensor_parallel_size becomes. Under vllm.yml's rules this is
# what `heads`, `kv_heads` and `paged_kv_heads` all map onto, so it shards the
# KV pool as well as the projections.
TENSOR_AXIS = "model"


def _call_maxtext_layout(function, config: Any, *args):
    """Call either generation of MaxText's layout-builder API.

    Chao's paged-KV branch originally inferred sharding from ``mesh`` alone.
    The config-aware API fixes that for serving recipes whose tensor axis is
    named ``model``. Supporting both keeps the TP=1 plugin usable while that
    companion MaxText change is reviewed.
    """
    if "config" in inspect.signature(function).parameters:
        return function(config, *args)
    return function(*args)


def build_mesh(tensor_parallel_size: int, devices: Any | None = None) -> Any:
    """Build the serving mesh, with `tensor_parallel_size` on the tensor axis.

    Args:
      tensor_parallel_size: devices to shard heads across.
      devices: JAX devices to use, defaulting to every device this process sees.
        Passed explicitly by tests and by any caller that has already decided on
        a device order.
    """
    import jax
    from jax.sharding import Mesh
    import numpy as np

    if tensor_parallel_size <= 0:
        raise ValueError(
            f"tensor_parallel_size must be positive, got {tensor_parallel_size}"
        )

    available = list(devices) if devices is not None else jax.devices()
    if len(available) < tensor_parallel_size:
        raise ValueError(
            f"tensor_parallel_size={tensor_parallel_size} needs that many devices "
            f"but JAX sees {len(available)}: {available}"
        )

    # Taking a prefix rather than all of them. A host with more GPUs than the
    # request asked for should serve on the requested number, not silently widen
    # the mesh and change the sharding.
    selected = available[:tensor_parallel_size]
    shape = (1, 1, tensor_parallel_size, 1, 1)
    mesh = Mesh(np.asarray(selected).reshape(shape), MESH_AXES)
    logger.info(
        "Serving mesh: %s over %d device(s) %s.",
        dict(mesh.shape),
        len(selected),
        [d.id for d in selected],
    )
    return mesh


def kv_head_shards(config: Any, mesh: Any) -> int:
    """How many ways the mesh splits KV heads.

    Delegates to MaxText so the pool and the kernels reading it cannot disagree
    about which axes are sharded, which would be wrong in a way no shape check
    would catch. `config` is MaxText's config: the axis names it resolves are
    per-model, so the mesh alone does not determine the answer.
    """
    from maxtext.inference.kv_execution.layout_builder import kv_head_shards as maxtext_shards

    return _call_maxtext_layout(maxtext_shards, config, mesh)


def pool_replicas(config: Any, mesh: Any) -> int:
    """Devices holding a copy of each KV-head shard."""
    from maxtext.inference.kv_execution.layout_builder import pool_replicas as maxtext_replicas

    return _call_maxtext_layout(maxtext_replicas, config, mesh)


def assert_kv_heads_shardable(config: Any, mesh: Any, num_kv_heads: int) -> None:
    """Refuse a mesh that cannot shard this model's KV heads evenly.

    Grouped-query models have few KV heads -- eight, for Llama-3.3-70B -- so a
    wide tensor-parallel mesh runs out of heads to distribute long before it
    runs out of devices. MaxText answers that by replicating, which is correct
    but multiplies the pool's footprint, so it should be a decision rather than
    a surprise discovered from an OOM.
    """
    shards = kv_head_shards(config, mesh)
    if shards <= 1 or not num_kv_heads:
        return
    if shards <= num_kv_heads and num_kv_heads % shards:
        raise ValueError(
            f"the mesh shards KV heads {shards} ways but the model has "
            f"{num_kv_heads}, which does not divide evenly"
        )
    if shards > num_kv_heads:
        logger.warning(
            "The mesh shards KV heads %d ways but the model has only %d, so each "
            "head is replicated %d times and the pool costs that much more.",
            shards,
            num_kv_heads,
            shards // num_kv_heads,
        )
