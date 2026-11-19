# SPDX-License-Identifier: Apache-2.0
"""KV cache spec for vLLM, and the pool layout derived from it.

Two vocabularies describe the same memory. vLLM negotiates in `KVCacheSpec` per
layer name and hands back a block count; MaxText allocates from a
`KvStorageLayoutV1`. This module keeps the translation in one place.

A convenient alignment worth stating, because it looks like a coincidence and is
relied upon: vLLM pops block id 0 at startup as a *null block*, flagged
`is_null`, never written, and therefore always reading as zero. That is exactly
MaxText's `padding_page_id=0` convention, so the reserved padding page needs no
special arrangement -- the two systems independently reserve the same page.
"""

from __future__ import annotations

from typing import Any

from jax_vllm_gpu.kv.common import KvStorageLayoutV1


def layer_names(num_layers: int) -> tuple[str, ...]:
    """Names for the attention layers vLLM will key its spec by.

    MaxText does not register vLLM `Attention` modules, so nothing supplies
    these; they are ours to choose and only have to be stable. The dotted form
    matches what vLLM's own models produce, which keeps its logs legible.
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    return tuple(f"model.layers.{i}.self_attn.attn" for i in range(num_layers))


def build_kv_cache_spec(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_size: int,
    dtype: Any,
    block_size: int,
) -> dict[str, Any]:
    """One `FullAttentionSpec` per layer, which is what a uniform-precision
    full-attention model needs and all this platform serves."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    return {
        name: FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
        )
        for name in layer_names(num_layers)
    }


def build_layout(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_name: str,
    tokens_per_page: int,
    num_pages: int,
    kv_head_shards: int = 1,
    pool_replicas: int = 1,
) -> KvStorageLayoutV1:
    """Build the pool layout from vLLM's negotiated block count.

    `num_pages` is taken as vLLM's total block count rather than that count plus
    one. MaxText's own builder adds a page because it reserves the padding page
    itself; here vLLM has already reserved block 0.
    """
    return KvStorageLayoutV1(
        tokens_per_page=tokens_per_page,
        num_pages=num_pages,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype_name,
        kv_head_shards=kv_head_shards,
        pool_replicas=pool_replicas,
        padding_page_id=0,
    )


def blocks_that_fit(layout: KvStorageLayoutV1, available_bytes: int) -> int:
    """How many pages fit in `available_bytes`, for one shard of the pool.

    vLLM asks a worker for bytes and computes the block count itself, so this
    exists for the reverse check: confirming the count it chose is one the pool
    can actually hold once replication is accounted for.
    """
    if available_bytes < 0:
        raise ValueError(f"available_bytes must not be negative, got {available_bytes}")
    per_page = layout.bytes_per_page() * 2 * layout.num_layers
    if per_page <= 0:
        raise ValueError("layout describes zero-sized pages")
    return available_bytes // per_page
