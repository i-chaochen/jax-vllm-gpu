# SPDX-License-Identifier: Apache-2.0
"""KV cache spec and the pool layout derived from it."""

from __future__ import annotations

import pytest

pytest.importorskip("vllm", reason="vLLM is supplied by the runtime image")

import torch

from jax_vllm_gpu.plugin.kv_spec import (
    blocks_that_fit,
    build_kv_cache_spec,
    build_layout,
    layer_names,
)


def test_layer_names_are_stable_and_one_per_layer():
    names = layer_names(3)
    assert names == (
        "model.layers.0.self_attn.attn",
        "model.layers.1.self_attn.attn",
        "model.layers.2.self_attn.attn",
    )
    assert layer_names(3) == names


def test_the_spec_has_one_entry_per_layer_and_vllm_can_size_it():
    spec = build_kv_cache_spec(
        num_layers=4,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
        block_size=16,
    )

    assert len(spec) == 4
    one = next(iter(spec.values()))
    # 16 tokens * 8 heads * 128 dim * 2 bytes * (K and V)
    assert one.page_size_bytes == 16 * 8 * 128 * 2 * 2


def test_the_layout_takes_vllms_block_count_without_adding_a_padding_page():
    """MaxText's own builder adds one because it reserves the padding page.
    vLLM already reserves block 0 as a null block, so adding another would
    silently shrink usable capacity by a page."""
    layout = build_layout(
        num_layers=4,
        num_kv_heads=8,
        head_dim=128,
        dtype_name="bfloat16",
        tokens_per_page=16,
        num_pages=100,
    )

    assert layout.num_pages == 100
    assert layout.padding_page_id == 0
    assert layout.max_tokens() == 99 * 16


def test_pool_bytes_account_for_head_sharding_and_replication():
    layout = build_layout(
        num_layers=2,
        num_kv_heads=8,
        head_dim=128,
        dtype_name="bfloat16",
        tokens_per_page=16,
        num_pages=10,
        kv_head_shards=4,
        pool_replicas=2,
    )

    assert layout.heads_per_shard() == 2
    assert layout.total_pool_bytes() == layout.pool_bytes_per_shard() * 4 * 2


def test_blocks_that_fit_is_the_inverse_of_the_per_page_cost():
    layout = build_layout(
        num_layers=2,
        num_kv_heads=8,
        head_dim=128,
        dtype_name="bfloat16",
        tokens_per_page=16,
        num_pages=1,
    )
    per_page = layout.bytes_per_page() * 2 * layout.num_layers

    assert blocks_that_fit(layout, per_page * 7) == 7
    assert blocks_that_fit(layout, per_page * 7 + 1) == 7
    assert blocks_that_fit(layout, 0) == 0


def test_zero_layers_is_rejected_rather_than_producing_an_empty_spec():
    with pytest.raises(ValueError, match="must be positive"):
        layer_names(0)
