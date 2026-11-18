# SPDX-License-Identifier: Apache-2.0
"""The metadata contract with MaxText's existing vLLM attention seam.

The value of these tests is that they exercise MaxText's own converter rather
than a copy of it, so a drift in that seam fails here rather than at a kernel
launch.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("maxtext.inference.kv_common", reason="MaxText must be importable")

from maxtext.layers.gpu_paged_attention import is_vllm_metadata

from jax_vllm_gpu.plugin.attention_metadata import build_attention_metadata
from jax_vllm_gpu.plugin.page_table import RequestPages, build_page_table

PAGE = 16


def table_for(requests):
    return build_page_table(requests, tokens_per_page=PAGE)


def test_maxtext_recognises_our_metadata_as_vllm_shaped():
    """If this fails, MaxText routes the step down its neutral-page-table branch
    instead, and the mismatch surfaces far from here."""
    md = build_attention_metadata(table_for([RequestPages("a", [1, 2], 0, 20)]))

    assert is_vllm_metadata(md)


def test_block_tables_are_padded_with_the_reserved_page():
    """Padding with the reserved page means an over-read addresses a page that
    reads as zero rather than another request's KV."""
    md = build_attention_metadata(
        table_for(
            [
                RequestPages("long", [1, 2, 3], 0, 40),
                RequestPages("short", [7], 0, 8),
            ]
        )
    )

    assert md.block_tables.shape == (2, 3)
    np.testing.assert_array_equal(md.block_tables[0], [1, 2, 3])
    np.testing.assert_array_equal(md.block_tables[1], [7, 0, 0])


def test_query_start_loc_is_the_exclusive_prefix_sum_of_query_lengths():
    md = build_attention_metadata(
        table_for(
            [
                RequestPages("a", [1, 2], 0, 18),
                RequestPages("b", [5], 0, 4),
            ]
        )
    )

    np.testing.assert_array_equal(md.query_start_loc, [0, 18, 22])


def test_input_positions_are_absolute_so_rope_matches_decode():
    md = build_attention_metadata(
        table_for([RequestPages("a", [1, 2], num_computed_tokens=20, num_scheduled_tokens=1)])
    )

    np.testing.assert_array_equal(md.input_positions, [20])


def test_a_row_narrower_than_a_request_needs_is_rejected():
    with pytest.raises(ValueError, match="too small"):
        build_attention_metadata(
            table_for([RequestPages("a", [1, 2, 3], 0, 40)]), max_blocks_per_seq=2
        )


def test_the_two_conversion_routes_agree_on_slot_mapping():
    """MaxText has two converters -- plan_from_neutral for its own page table and
    plan_from_vllm for vLLM-shaped metadata. We produce both from one source, so
    they must agree; a divergence would mean tokens written to the wrong slots
    under one frontend and not the other."""
    import jax

    requests = [
        RequestPages("a", [1, 2], 0, 18),
        RequestPages("b", [5], 0, 4),
    ]
    table = table_for(requests)
    md = build_attention_metadata(table)

    from maxtext.layers.gpu_paged_attention import plan_from_neutral, plan_from_vllm

    with jax.default_device(jax.devices("cpu")[0]):
        neutral = plan_from_neutral(table, PAGE)
        vllm_plan = plan_from_vllm(
            md,
            PAGE,
            total_tokens=int(table.num_tokens),
            max_seqlen_k=int(table.seq_lens.max()),
        )

        np.testing.assert_array_equal(
            np.asarray(neutral.slot_mapping), np.asarray(vllm_plan.slot_mapping)
        )
        np.testing.assert_array_equal(
            np.asarray(neutral.cu_seqlens_q), np.asarray(vllm_plan.cu_seqlens_q)
        )

        # kv_indptr is the contract: it delimits each request's run of pages, so
        # agreement here plus agreement on the delimited prefix is what the
        # kernels actually depend on.
        neutral_indptr = np.asarray(neutral.kv_indptr)
        vllm_indptr = np.asarray(vllm_plan.kv_indptr)
        np.testing.assert_array_equal(neutral_indptr, vllm_indptr)

        used = int(neutral_indptr[-1])
        np.testing.assert_array_equal(
            np.asarray(neutral.kv_page_indices)[:used],
            np.asarray(vllm_plan.kv_page_indices)[:used],
        )


def test_the_vllm_route_leaves_trailing_padding_in_kv_page_indices():
    """Documenting a real difference between MaxText's two converters, not
    asserting a preference. plan_from_vllm sizes kv_page_indices as
    num_seqs * max_blocks_per_seq, so a ragged batch leaves unused trailing
    entries, while plan_from_neutral emits exactly the pages needed. It is inert
    because kv_indptr bounds the read, but anyone comparing the two arrays
    directly will see a shape mismatch and think something is broken."""
    import jax

    requests = [
        RequestPages("a", [1, 2], 0, 18),
        RequestPages("b", [5], 0, 4),
    ]
    table = table_for(requests)
    md = build_attention_metadata(table)

    from maxtext.layers.gpu_paged_attention import plan_from_neutral, plan_from_vllm

    with jax.default_device(jax.devices("cpu")[0]):
        neutral = plan_from_neutral(table, PAGE)
        vllm_plan = plan_from_vllm(
            md, PAGE, total_tokens=int(table.num_tokens), max_seqlen_k=int(table.seq_lens.max())
        )

    used = int(np.asarray(neutral.kv_indptr)[-1])
    assert np.asarray(neutral.kv_page_indices).size == used
    assert np.asarray(vllm_plan.kv_page_indices).size >= used
