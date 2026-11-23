# SPDX-License-Identifier: Apache-2.0
"""The per-step pipeline, and the ordering the guarantee depends on."""

from __future__ import annotations

import numpy as np
import pytest


from vllm_xla_gpu.kv.execution.bucketing import StepShapePlanner

from vllm_xla_gpu.plugin.kv_reuse import PageReuseTracker
from vllm_xla_gpu.plugin.page_table import RequestPages
from vllm_xla_gpu.plugin.step import prepare_step

PAGE = 16
POOL = 64


def planner() -> StepShapePlanner:
    return StepShapePlanner(
        tokens_per_page=PAGE,
        max_batch=8,
        max_context_len=512,
        pool_pages=POOL,
    )


def tokens_for(request: RequestPages) -> np.ndarray:
    return np.arange(request.num_scheduled_tokens, dtype=np.int32)


def test_a_prefill_step_prepares_a_view_and_inputs():
    tracker = PageReuseTracker(num_pages=POOL, tokens_per_page=PAGE)
    requests = [RequestPages("a", [1, 2], 0, 20)]
    shape = planner().extend_shape(num_tokens=20, max_seq_len=20, num_requests=1)

    prepared = prepare_step(
        requests,
        tracker=tracker,
        scrub=lambda pages: None,
        shape=shape,
        tokens_per_page=PAGE,
        token_ids=tokens_for,
    )

    assert prepared.view.num_active_requests == 1
    assert prepared.view.num_active_tokens == 20
    assert prepared.scrubbed.size == 0
    np.testing.assert_array_equal(prepared.table.query_lens, [20])


def test_a_recycled_page_is_scrubbed_before_the_table_is_built():
    """The ordering test. If the table were built first, assert_clean would
    raise, so a passing run proves the scrub happened in between."""
    tracker = PageReuseTracker(num_pages=POOL, tokens_per_page=PAGE)
    first = [RequestPages("a", [1, 2], 0, 32)]
    shape = planner().extend_shape(num_tokens=32, max_seq_len=32, num_requests=1)
    prepare_step(
        first,
        tracker=tracker,
        scrub=lambda pages: None,
        shape=shape,
        tokens_per_page=PAGE,
        token_ids=tokens_for,
    )

    scrubbed: list[list[int]] = []
    second = [RequestPages("b", [1, 3], 0, 20)]
    prepared = prepare_step(
        second,
        tracker=tracker,
        scrub=lambda pages: scrubbed.append(sorted(int(p) for p in pages)),
        shape=planner().extend_shape(num_tokens=20, max_seq_len=20, num_requests=1),
        tokens_per_page=PAGE,
        token_ids=tokens_for,
    )

    assert scrubbed == [[1]]
    np.testing.assert_array_equal(prepared.scrubbed, [1])
    assert tracker.num_pending == 0


def test_a_scrub_that_does_nothing_still_leaves_no_obligation():
    """confirm_scrubbed is driven by the pipeline rather than by the scrub
    callback, so a no-op device scrub cannot leave a stale obligation that
    blocks every later step."""
    tracker = PageReuseTracker(num_pages=POOL, tokens_per_page=PAGE)
    shape = planner().extend_shape(num_tokens=16, max_seq_len=16, num_requests=1)
    prepare_step(
        [RequestPages("a", [1], 0, 16)],
        tracker=tracker,
        scrub=lambda pages: None,
        shape=shape,
        tokens_per_page=PAGE,
        token_ids=tokens_for,
    )
    prepare_step(
        [RequestPages("b", [1], 0, 16)],
        tracker=tracker,
        scrub=lambda pages: None,
        shape=shape,
        tokens_per_page=PAGE,
        token_ids=tokens_for,
    )

    assert tracker.num_pending == 0


def test_a_prefix_cache_hit_triggers_no_scrub():
    tracker = PageReuseTracker(num_pages=POOL, tokens_per_page=PAGE)
    shape = planner().extend_shape(num_tokens=32, max_seq_len=32, num_requests=1)
    prepare_step(
        [RequestPages("a", [1, 2], 0, 32)],
        tracker=tracker,
        scrub=lambda pages: None,
        shape=shape,
        tokens_per_page=PAGE,
        token_ids=tokens_for,
    )

    calls: list[np.ndarray] = []
    prepared = prepare_step(
        [RequestPages("b", [1, 2, 5], 32, 4)],
        tracker=tracker,
        scrub=calls.append,
        shape=planner().extend_shape(num_tokens=4, max_seq_len=36, num_requests=1),
        tokens_per_page=PAGE,
        token_ids=tokens_for,
    )

    assert calls == []
    assert prepared.scrubbed.size == 0


def test_a_decode_step_feeds_one_absolute_position_per_request():
    tracker = PageReuseTracker(num_pages=POOL, tokens_per_page=PAGE)
    requests = [
        RequestPages("a", [1, 2], 20, 1),
        RequestPages("b", [3], 5, 1),
    ]
    shape = planner().decode_shape(num_requests=2, max_seq_len=21)

    prepared = prepare_step(
        requests,
        tracker=tracker,
        scrub=lambda pages: None,
        shape=shape,
        tokens_per_page=PAGE,
        token_ids=tokens_for,
    )

    np.testing.assert_array_equal(prepared.table.write_positions, [20, 5])
    assert prepared.inputs.segment_ids is None


def test_a_token_count_that_disagrees_with_the_schedule_is_rejected():
    tracker = PageReuseTracker(num_pages=POOL, tokens_per_page=PAGE)
    shape = planner().extend_shape(num_tokens=20, max_seq_len=20, num_requests=1)

    with pytest.raises(ValueError, match="token_ids yielded"):
        prepare_step(
            [RequestPages("a", [1, 2], 0, 20)],
            tracker=tracker,
            scrub=lambda pages: None,
            shape=shape,
            tokens_per_page=PAGE,
            token_ids=lambda r: np.zeros(3, dtype=np.int32),
        )


def test_a_failed_step_does_not_record_its_pages_as_written():
    """note_written runs last. If it ran earlier, a step that raised would leave
    pages marked as owned and suppress a later, genuine scrub."""
    tracker = PageReuseTracker(num_pages=POOL, tokens_per_page=PAGE)
    shape = planner().extend_shape(num_tokens=20, max_seq_len=20, num_requests=1)

    with pytest.raises(ValueError):
        prepare_step(
            [RequestPages("a", [1, 2], 0, 20)],
            tracker=tracker,
            scrub=lambda pages: None,
            shape=shape,
            tokens_per_page=PAGE,
            token_ids=lambda r: np.zeros(1, dtype=np.int32),
        )

    assert tracker.observe([RequestPages("b", [1, 2], 0, 20)]).size == 0
