# SPDX-License-Identifier: Apache-2.0
"""Translation from vLLM block ids to MaxText's page table."""

from __future__ import annotations

import numpy as np
import pytest


from vllm_xla_gpu.plugin.page_table import RequestPages, build_page_table

PAGE = 16


def test_a_fresh_prefill_writes_from_position_zero():
    table = build_page_table(
        [RequestPages("a", [3, 4], num_computed_tokens=0, num_scheduled_tokens=20)],
        tokens_per_page=PAGE,
    )

    assert table.page_ids == [[3, 4]]
    np.testing.assert_array_equal(table.seq_lens, [20])
    np.testing.assert_array_equal(table.query_lens, [20])
    np.testing.assert_array_equal(table.write_positions, np.arange(20))


def test_decode_writes_a_single_absolute_position():
    """The position must be the next absolute index, not zero. Getting this
    wrong applies RoPE at the wrong offset and is invisible without a
    token-identical reference."""
    table = build_page_table(
        [RequestPages("a", [3, 4], num_computed_tokens=20, num_scheduled_tokens=1)],
        tokens_per_page=PAGE,
    )

    np.testing.assert_array_equal(table.write_positions, [20])
    np.testing.assert_array_equal(table.seq_lens, [21])
    np.testing.assert_array_equal(table.query_lens, [1])


def test_a_prefix_cache_hit_skips_the_cached_positions():
    """A hit raises num_computed_tokens without scheduling those tokens, so the
    step must write only the uncached suffix while still addressing every page
    that holds context."""
    table = build_page_table(
        [RequestPages("a", [7, 8, 9], num_computed_tokens=32, num_scheduled_tokens=5)],
        tokens_per_page=PAGE,
    )

    assert table.page_ids == [[7, 8, 9]]
    np.testing.assert_array_equal(table.write_positions, np.arange(32, 37))
    np.testing.assert_array_equal(table.seq_lens, [37])


def test_over_allocated_block_lists_are_truncated_to_the_context():
    """vLLM may hand over more blocks than the context occupies. Passing them
    through unmodified fails KvPageTableV1.validate, which requires exactly
    ceil(seq_len / page) pages."""
    table = build_page_table(
        [RequestPages("a", [1, 2, 3, 4, 5], num_computed_tokens=0, num_scheduled_tokens=17)],
        tokens_per_page=PAGE,
    )

    assert table.page_ids == [[1, 2]]


def test_too_few_blocks_is_an_error_rather_than_a_short_read():
    with pytest.raises(ValueError, match="needs 2 pages"):
        build_page_table(
            [RequestPages("a", [1], num_computed_tokens=0, num_scheduled_tokens=17)],
            tokens_per_page=PAGE,
        )


def test_a_mixed_batch_concatenates_positions_in_request_order():
    table = build_page_table(
        [
            RequestPages("prefill", [1, 2], num_computed_tokens=0, num_scheduled_tokens=18),
            RequestPages("decode", [5], num_computed_tokens=9, num_scheduled_tokens=1),
        ],
        tokens_per_page=PAGE,
    )

    np.testing.assert_array_equal(table.query_lens, [18, 1])
    np.testing.assert_array_equal(
        table.write_positions, np.concatenate([np.arange(18), [9]])
    )
    np.testing.assert_array_equal(table.request_order, [0, 1])


def test_last_page_lens_stay_exact_so_kernels_cannot_read_a_recycled_tail():
    """Exactness here is one of the three things the recycled-page guarantee
    rests on: a rounded-up length would expose whatever the page held before."""
    table = build_page_table(
        [RequestPages("a", [1, 2], num_computed_tokens=0, num_scheduled_tokens=17)],
        tokens_per_page=PAGE,
    )

    np.testing.assert_array_equal(table.last_page_lens(PAGE), [1])


def test_a_request_with_nothing_scheduled_is_rejected():
    with pytest.raises(ValueError, match="no scheduled tokens"):
        build_page_table(
            [RequestPages("a", [1], num_computed_tokens=5, num_scheduled_tokens=0)],
            tokens_per_page=PAGE,
        )
