# SPDX-License-Identifier: Apache-2.0
"""The physical KV pool, allocated from the block count vLLM negotiated.

vLLM decides *how many* blocks exist; MaxText decides what a block looks like in
memory. This module is where the two meet: it takes the negotiated count, builds
a `KvStorageLayoutV1` around it, and allocates through MaxText's `allocate_pool`
so the arrays are the same ones the kernels are written against.

Allocation goes through JAX, never `torch.zeros`. torch here is a CPU-only build
with no HIP runtime at all, so it could not allocate device memory even if asked;
more to the point the pool has to carry a JAX sharding for the kernels to run
under `shard_map`.

**The pool is donated into every step and comes back aliased.** The arrays handed
in are invalid afterwards, and only the returned handles refer to live memory, so
`rebind` is not an optimisation -- skipping it means the next step reads a deleted
buffer.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from jax_vllm_gpu.kv.common import KvStorageLayoutV1

logger = logging.getLogger(__name__)


def allocate(layout: KvStorageLayoutV1, mesh: Any | None = None) -> Any:
    """Allocate a zeroed pool for `layout`, sharded to `mesh`.

    Zeroed rather than uninitialised, and that matters twice over: vLLM's block
    0 is a null block promised to read as zero, and the recycled-page tracker
    treats a never-written page as clean, which is only true if it starts empty.
    """
    from jax_vllm_gpu.kv.execution.layout_builder import kv_pool_sharding
    from jax_vllm_gpu.kv.execution.pool_factory import allocate_pool

    sharding = kv_pool_sharding(mesh, layout)
    pool = allocate_pool(layout, sharding=sharding)
    logger.info(
        "Allocated KV pool: %d layers x 2 x %s %s, %.2f GiB per shard across %d shard(s).",
        layout.num_layers,
        pool.page_shape,
        layout.dtype,
        layout.pool_bytes_per_shard() / 2**30,
        layout.kv_head_shards,
    )
    return pool


def as_kv_caches(pool: Any) -> list[list[Any]]:
    """The pool in the nested shape MaxText's decoder distributes per layer.

    `kv_caches[i]` reaches layer `i` as its `kv_cache`, and the `gpu_paged`
    branch of `Attention.__call__` unpacks that as `[k_pool, v_pool]`.
    """
    return [[pool.k_pages[i], pool.v_pages[i]] for i in range(pool.num_layers)]


def rebind(pool: Any, updated: Any) -> None:
    """Store the aliased handles a step returned, in place."""
    for layer, pair in enumerate(updated):
        pool.replace_layer(layer, pair[0], pair[1])


def make_scrub_fn(pool: Any):
    """A `step.ScrubFn` that zeroes pages of this pool, in one dispatch.

    Closes over the pool rather than taking it as an argument because
    `prepare_step` deliberately knows nothing about device memory; the scrub is
    injected precisely so the host-side ordering stays testable without a GPU.
    """
    from jax_vllm_gpu.kv.execution.pool_ops import scrub_pages_all_layers

    def scrub(page_ids: np.ndarray) -> None:
        if page_ids.size == 0:
            return
        k_pages, v_pages = scrub_pages_all_layers(pool.k_pages, pool.v_pages, page_ids)
        for layer in range(pool.num_layers):
            pool.replace_layer(layer, k_pages[layer], v_pages[layer])

    return scrub
