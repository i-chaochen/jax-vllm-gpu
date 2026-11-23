# SPDX-License-Identifier: Apache-2.0
"""M4's poisoned-page test, carried across to EXTERNAL_VLLM mode.

The guarantee M4 delivered lives entirely inside the components this mode
bypasses: the dirty bitmap on `PagedBlockAllocator`, `pending_scrub` and
`confirm_scrubbed` on `NativeKvControlPlane`, and the refusal inside
`build_page_table`. Handing page ownership to vLLM removes all three. So this is
not a re-run of an existing test against new plumbing -- it is the evidence that
the property survived the change of page owner rather than being dropped along
with the allocator that used to provide it.

vLLM does not close the gap itself. `get_new_blocks` resets a block's hash and
nothing else, and `new_block_ids_to_zero` is populated only for Mamba or
mixed-precision caches, so a uniform-precision attention deployment gets no
zeroing at all.

Poison rather than zeros as the donor's mark, for M4's reason: a page that was
never scrubbed and a page that was scrubbed then written both contain plausible
numbers, and only a sentinel tells them apart.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax", reason="the pool is device memory")

import jax.numpy as jnp

from vllm_xla_gpu.kv.common import KvStorageLayoutV1
from vllm_xla_gpu.host.shapes import StepShape
from vllm_xla_gpu.kv.execution.pool_factory import allocate_pool
from vllm_xla_gpu.kv.execution.pool_ops import POISON_SENTINEL, poison_pages

from vllm_xla_gpu.plugin import pool as pool_module
from vllm_xla_gpu.plugin.kv_reuse import DirtyPageError, PageReuseTracker
from vllm_xla_gpu.plugin.page_table import RequestPages, build_page_table
from vllm_xla_gpu.plugin.step import prepare_step

PAGE = 16
NUM_PAGES = 8
READER_VALUE = 3.0


def a_pool():
    layout = KvStorageLayoutV1(
        tokens_per_page=PAGE,
        num_pages=NUM_PAGES,
        num_layers=1,
        num_kv_heads=2,
        head_dim=4,
        dtype="bfloat16",
        padding_page_id=0,
    )
    return layout, allocate_pool(layout)


def a_shape(num_requests, num_tokens):
    return StepShape(
        num_requests=num_requests,
        num_tokens=num_tokens,
        num_pages=NUM_PAGES,
        max_seqlen_k=PAGE * 2,
        is_decode=num_tokens == num_requests,
    )


def contaminate(pool, page):
    """Leave a previous occupant's KV in `page`, the way a freed block does."""
    k, v = poison_pages(pool.k_pages[0], pool.v_pages[0], np.asarray([page], np.int32))
    pool.replace_layer(0, k, v)


def page_contents(pool, page):
    return np.asarray(pool.k_pages[0].astype(jnp.float32))[page]


def donor_then_reader(tracker, donor_page):
    """The donor writes `donor_page`, finishes, and a reader is handed it back.

    This is the only shape in which a recycled block can reach a request: cache
    hits arrive through `BlockPool.touch()`, never `get_new_blocks()`, so a page
    another request wrote can only ever appear in the *write* region.
    """
    donor = RequestPages(
        request_id="donor",
        block_ids=[donor_page],
        num_computed_tokens=0,
        num_scheduled_tokens=PAGE,
    )
    tracker.observe([donor])
    tracker.note_written([donor])

    return RequestPages(
        request_id="reader",
        block_ids=[donor_page],
        num_computed_tokens=0,
        num_scheduled_tokens=1,
    )


def test_a_page_another_request_wrote_is_flagged_when_it_comes_back():
    tracker = PageReuseTracker(num_pages=NUM_PAGES, tokens_per_page=PAGE)
    reader = donor_then_reader(tracker, donor_page=3)

    assert list(tracker.observe([reader])) == [3]


def test_a_page_the_same_request_already_owns_is_not_flagged():
    """Re-zeroing a request's own partially filled page would wipe the valid
    prefix it is about to read."""
    tracker = PageReuseTracker(num_pages=NUM_PAGES, tokens_per_page=PAGE)
    mine = RequestPages(
        request_id="mine",
        block_ids=[3, 4],
        num_computed_tokens=0,
        num_scheduled_tokens=PAGE,
    )
    tracker.observe([mine])
    tracker.note_written([mine])

    continuing = RequestPages(
        request_id="mine",
        block_ids=[3, 4],
        num_computed_tokens=PAGE,
        num_scheduled_tokens=1,
    )

    assert tracker.observe([continuing]).size == 0


def test_a_prefix_cache_hit_is_not_mistaken_for_a_recycled_page():
    """A hit lands in the *read* region, `block_ids[:num_computed // page]`.
    Scrubbing it would destroy both the hit and correctness."""
    tracker = PageReuseTracker(num_pages=NUM_PAGES, tokens_per_page=PAGE)
    donor_then_reader(tracker, donor_page=3)

    hit = RequestPages(
        request_id="reader",
        block_ids=[3, 4],
        num_computed_tokens=PAGE,
        num_scheduled_tokens=1,
    )

    assert tracker.observe([hit]).size == 0


def test_the_page_table_is_refused_while_the_page_is_still_dirty():
    """The structural enforcement point, and the analogue of M4's
    `DirtyPageError` from `build_page_table`."""
    tracker = PageReuseTracker(num_pages=NUM_PAGES, tokens_per_page=PAGE)
    reader = donor_then_reader(tracker, donor_page=3)
    tracker.observe([reader])

    table = build_page_table([reader], tokens_per_page=PAGE)
    with pytest.raises(DirtyPageError):
        tracker.assert_clean(table.flat_page_indices())


def test_the_step_pipeline_scrubs_before_it_builds_the_table():
    """The order is what carries the guarantee. `prepare_step` scrubs, confirms,
    then builds -- so a step that completes cannot have described a dirty page."""
    _, pool = a_pool()
    tracker = PageReuseTracker(num_pages=NUM_PAGES, tokens_per_page=PAGE)
    reader = donor_then_reader(tracker, donor_page=3)
    contaminate(pool, 3)

    prepared = prepare_step(
        [reader],
        tracker=tracker,
        scrub=pool_module.make_scrub_fn(pool),
        shape=a_shape(1, 1),
        tokens_per_page=PAGE,
        token_ids=lambda request: np.zeros((1,), np.int32),
    )

    assert list(prepared.scrubbed) == [3]
    assert tracker.num_pending == 0


def test_no_recycled_byte_survives_in_the_page_the_reader_is_given():
    """The property the milestone asks for, over the whole page rather than only
    the readable extent -- an over-read is exactly what the scrub exists for."""
    _, pool = a_pool()
    tracker = PageReuseTracker(num_pages=NUM_PAGES, tokens_per_page=PAGE)
    reader = donor_then_reader(tracker, donor_page=3)
    contaminate(pool, 3)

    prepare_step(
        [reader],
        tracker=tracker,
        scrub=pool_module.make_scrub_fn(pool),
        shape=a_shape(1, 1),
        tokens_per_page=PAGE,
        token_ids=lambda request: np.zeros((1,), np.int32),
    )

    written = page_contents(pool, 3)
    assert not (written == POISON_SENTINEL).any(), "the donor's marker survived"
    assert (written == 0.0).all(), "the page must be entirely zero before reuse"


def test_without_the_scrub_the_contamination_is_still_there():
    """The negative control. Without it the assertion above passes trivially on
    a pool that was simply never dirtied."""
    _, pool = a_pool()
    contaminate(pool, 3)

    written = page_contents(pool, 3)
    assert (written == POISON_SENTINEL).all()


def test_the_padding_page_reads_as_zero_so_an_over_read_is_harmless():
    """vLLM reserves block 0 as a null block it never writes, which is exactly
    MaxText's padding_page_id. Both systems independently rely on it reading
    zero, so it is worth asserting rather than assuming."""
    _, pool = a_pool()

    assert (page_contents(pool, 0) == 0.0).all()
