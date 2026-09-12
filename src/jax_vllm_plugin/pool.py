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
import os
from typing import Any

import numpy as np

from maxtext.inference.kv_common import KvStorageLayoutV1

logger = logging.getLogger(__name__)
FUSED_SCRUB_ENV_VAR = "JAX_VLLM_FUSED_PAGE_SCRUB"


def allocate(
    layout: KvStorageLayoutV1, mesh: Any | None = None, config: Any | None = None
) -> Any:
    """Allocate a zeroed pool for `layout`, sharded to `mesh`.

    Zeroed rather than uninitialised, and that matters twice over: vLLM's block
    0 is a null block promised to read as zero, and the recycled-page tracker
    treats a never-written page as clean, which is only true if it starts empty.
    """
    from maxtext.inference.kv_execution.layout_builder import kv_pool_sharding
    from maxtext.inference.kv_execution.pool_factory import allocate_pool
    from .mesh import _call_maxtext_layout

    sharding = _call_maxtext_layout(kv_pool_sharding, config, mesh, layout)
    pool = allocate_pool(layout, sharding=sharding)
    k_shape, v_shape = pool.page_shapes
    logger.info(
        "Allocated KV pool (%s): %d layers x K %s / V %s %s, "
        "%.2f GiB per shard across %d shard(s).",
        layout.kv_layout,
        layout.num_layers,
        k_shape,
        v_shape,
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


def scrub_bucket(size: int) -> int:
    """Round a scrub count up to a power of two.

    `jax.jit` specializes on `page_ids.shape[0]`, so an unbucketed count compiles
    a fresh executable for every distinct number of recycled pages: one measured
    replay accumulated 158 of them, 8 compiling *during* measurement and
    contaminating step preparation by ~222-227 ms while the launcher's gate, which
    watched only `jit(step)`, reported zero compilations.

    A power-of-two ladder bounds that at about a dozen, and matches the
    convention `pool_ops._pad_page_indices` already uses for the non-fused path.
    """
    size = int(size)
    if size <= 1:
        return 1
    return 1 << (size - 1).bit_length()


def make_scrub_fn(pool: Any):
    """A `step.ScrubFn` that zeroes pages of this pool.

    Closes over the pool rather than taking it as an argument because
    `prepare_step` deliberately knows nothing about device memory; the scrub is
    injected precisely so the host-side ordering stays testable without a GPU.
    """
    if os.environ.get(FUSED_SCRUB_ENV_VAR, "1") != "0":
        from jax_aiter.ffi.registry import standalone_symbol_available
        from jax_aiter.ops.scrub_kv_pages import TARGET, scrub_kv_pages

        if standalone_symbol_available(TARGET):
            import jax
            import jax.numpy as jnp

            fused_scrub = jax.jit(scrub_kv_pages, donate_argnums=(0,))

            def scrub(page_ids: np.ndarray) -> None:
                page_ids = np.asarray(page_ids, dtype=np.int32).reshape(-1)
                if page_ids.size == 0:
                    return
                width = scrub_bucket(page_ids.size)
                if width != page_ids.size:
                    # -1 is inert by construction: the first-party kernel rejects
                    # negative page ids before it computes any address
                    # (csrc/ffi/kv/scrub_kv_pages_ja.cu). Each pad entry costs one
                    # block per pool that returns immediately.
                    page_ids = np.concatenate(
                        [
                            page_ids,
                            np.full((width - page_ids.size,), -1, dtype=np.int32),
                        ]
                    )
                pools = tuple(pool.k_pages) + tuple(pool.v_pages)
                updated = fused_scrub(pools, jnp.asarray(page_ids, jnp.int32))
                for layer in range(pool.num_layers):
                    pool.replace_layer(
                        layer,
                        updated[layer],
                        updated[pool.num_layers + layer],
                    )

            def warm(max_pages: int) -> tuple[int, ...]:
                """Compile every rung, so measurement never pays for one.

                An all -1 batch is a real call at the rung's shape but scrubs
                nothing, so warming cannot disturb pool contents.
                """
                rungs = []
                width = 1
                while True:
                    scrub(np.full((width,), -1, dtype=np.int32))
                    rungs.append(width)
                    if width >= max_pages:
                        break
                    width <<= 1
                logger.info("Warmed %d page-scrub shapes: %s.", len(rungs), rungs)
                return tuple(rungs)

            scrub.warm = warm
            logger.info(
                "Using one-dispatch JAX-AITER page scrub for %d KV pools.",
                2 * pool.num_layers,
            )
            return scrub

        logger.warning(
            "%s is enabled but %s is unavailable; using the per-pool JAX scrub.",
            FUSED_SCRUB_ENV_VAR,
            TARGET,
        )

    from maxtext.inference.kv_execution.pool_ops import scrub_pages_all_layers

    def scrub(page_ids: np.ndarray) -> None:
        if page_ids.size == 0:
            return
        k_pages, v_pages = scrub_pages_all_layers(pool.k_pages, pool.v_pages, page_ids)
        for layer in range(pool.num_layers):
            pool.replace_layer(layer, k_pages[layer], v_pages[layer])

    return scrub
