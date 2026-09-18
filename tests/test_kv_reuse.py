# SPDX-License-Identifier: Apache-2.0
"""Recycled-page tracking under vLLM's block lifecycle.

The two failure modes are opposite and both silent, so each gets a test:
scrubbing too little leaks another request's KV, and scrubbing a prefix-cache
hit corrupts the request that was supposed to read it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("maxtext.inference.kv_common", reason="MaxText must be importable")

from jax_vllm_plugin.kv_reuse import DirtyPageError, PageReuseTracker
from jax_vllm_plugin.page_table import RequestPages

PAGE = 16
POOL = 64


def tracker() -> PageReuseTracker:
    return PageReuseTracker(num_pages=POOL, tokens_per_page=PAGE)


def test_a_never_used_pool_needs_no_scrubbing():
    """The pool is allocated with jnp.zeros, so untouched pages are clean."""
    t = tracker()
    pending = t.observe([RequestPages("a", [1, 2], 0, 20)])
    assert pending.size == 0


def test_a_page_recycled_from_another_request_must_be_scrubbed():
    t = tracker()
    first = [RequestPages("a", [1, 2], 0, 20)]
    t.observe(first)
    t.note_written(first)

    # vLLM frees a's blocks and hands page 1 to b as scratch.
    pending = t.observe([RequestPages("b", [1, 3], 0, 20)])

    np.testing.assert_array_equal(pending, [1])


def test_a_prefix_cache_hit_is_never_scrubbed():
    """The corruption case. b reads pages 1 and 2 as a cache hit, so
    num_computed_tokens covers them and they sit in the read region. Zeroing
    them would destroy both the hit and correctness."""
    t = tracker()
    first = [RequestPages("a", [1, 2], 0, 32)]
    t.observe(first)
    t.note_written(first)

    hit = [RequestPages("b", [1, 2, 5], num_computed_tokens=32, num_scheduled_tokens=4)]
    pending = t.observe(hit)

    assert pending.size == 0


def test_a_request_continuing_into_its_own_partial_page_is_not_rescrubbed():
    """Re-zeroing a page the same request already partly filled would wipe its
    own valid prefix, which is the carve-out vLLM documents too."""
    t = tracker()
    step_one = [RequestPages("a", [1], 0, 8)]
    t.observe(step_one)
    t.note_written(step_one)

    pending = t.observe([RequestPages("a", [1], num_computed_tokens=8, num_scheduled_tokens=4)])

    assert pending.size == 0


def test_a_reused_request_id_gets_a_fresh_page_owner_generation():
    """Request IDs are caller-controlled names, not stable ownership tokens."""
    t = tracker()
    t.admit("reused")
    first = [RequestPages("reused", [1], 0, 8)]
    t.observe(first)
    t.note_written(first)

    t.retire("reused")
    t.admit("reused")
    pending = t.observe([RequestPages("reused", [1], 0, 4)])

    np.testing.assert_array_equal(pending, [1])


def test_retirement_bounds_the_active_request_map_without_erasing_page_history():
    t = tracker()
    t.admit("done")
    first = [RequestPages("done", [1], 0, 8)]
    t.note_written(first)

    t.retire("done")

    assert t.num_active_requests == 0
    assert list(t.observe([RequestPages("next", [1], 0, 4)])) == [1]


def test_duplicate_active_request_id_is_rejected():
    t = tracker()
    t.admit("active")

    with pytest.raises(ValueError, match="already active"):
        t.admit("active")


def test_reading_the_obligation_does_not_discharge_it():
    """Only confirm_scrubbed clears, so a caller that forgets to zero on device
    cannot clear the record by looking at it."""
    t = tracker()
    first = [RequestPages("a", [1], 0, 16)]
    t.observe(first)
    t.note_written(first)
    t.observe([RequestPages("b", [1], 0, 16)])

    assert t.pending_scrub().size == 1
    assert t.pending_scrub().size == 1

    t.confirm_scrubbed([1])
    assert t.pending_scrub().size == 0


def test_a_page_table_naming_an_unscrubbed_page_is_refused():
    t = tracker()
    first = [RequestPages("a", [1], 0, 16)]
    t.observe(first)
    t.note_written(first)
    t.observe([RequestPages("b", [1], 0, 16)])

    with pytest.raises(DirtyPageError, match=r"\[1\]"):
        t.assert_clean([1])

    t.confirm_scrubbed([1])
    t.assert_clean([1])


def test_obligations_accumulate_across_observations():
    t = tracker()
    first = [RequestPages("a", [1, 2], 0, 32)]
    t.observe(first)
    t.note_written(first)

    t.observe([RequestPages("b", [1], 0, 16)])
    t.observe([RequestPages("c", [2], 0, 16)])

    np.testing.assert_array_equal(t.pending_scrub(), [1, 2])


def test_the_write_region_excludes_the_computed_prefix():
    t = tracker()
    request = RequestPages("a", [4, 5, 6], num_computed_tokens=32, num_scheduled_tokens=5)

    assert t.write_region(request) == [6]


def test_a_page_outside_the_pool_is_rejected():
    t = tracker()
    with pytest.raises(ValueError, match="out of range"):
        t.observe([RequestPages("a", [POOL + 1], 0, 16)])
