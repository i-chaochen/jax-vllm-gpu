# SPDX-License-Identifier: Apache-2.0
"""Accumulating vLLM's block-table deltas across steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

pytest.importorskip("maxtext.inference.kv_common", reason="MaxText must be importable")

from jax_vllm_plugin.batch_state import BatchState


@dataclass
class FakeNewReq:
    req_id: str
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int


@dataclass
class FakeCached:
    req_ids: list[str] = field(default_factory=list)
    resumed_req_ids: set[str] = field(default_factory=set)
    new_block_ids: list[tuple[list[int], ...] | None] = field(default_factory=list)
    num_computed_tokens: list[int] = field(default_factory=list)


@dataclass
class FakeOutput:
    scheduled_new_reqs: list[FakeNewReq] = field(default_factory=list)
    scheduled_cached_reqs: FakeCached | None = None
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    finished_req_ids: set[str] = field(default_factory=set)


def test_a_new_request_takes_its_whole_block_table():
    state = BatchState()
    state.apply(
        FakeOutput(
            scheduled_new_reqs=[FakeNewReq("a", ([1, 2],), 0)],
            num_scheduled_tokens={"a": 20},
        )
    )

    assert state.get("a").block_ids == [1, 2]
    assert state.get("a").num_computed_tokens == 0


def test_a_continuing_request_appends_only_the_new_blocks():
    state = BatchState()
    state.apply(FakeOutput(scheduled_new_reqs=[FakeNewReq("a", ([1, 2],), 0)]))
    state.apply(
        FakeOutput(
            scheduled_cached_reqs=FakeCached(
                req_ids=["a"], new_block_ids=[([3],)], num_computed_tokens=[32]
            )
        )
    )

    assert state.get("a").block_ids == [1, 2, 3]
    assert state.get("a").num_computed_tokens == 32


def test_a_resumed_request_replaces_its_block_table():
    """vLLM may have freed and reallocated a preempted request's blocks, so
    appending would name pages it no longer owns."""
    state = BatchState()
    state.apply(FakeOutput(scheduled_new_reqs=[FakeNewReq("a", ([1, 2],), 0)]))
    state.apply(
        FakeOutput(
            scheduled_cached_reqs=FakeCached(
                req_ids=["a"],
                resumed_req_ids={"a"},
                new_block_ids=[([7, 8, 9],)],
                num_computed_tokens=[20],
            )
        )
    )

    assert state.get("a").block_ids == [7, 8, 9]


def test_a_step_with_no_new_blocks_still_updates_the_computed_count():
    state = BatchState()
    state.apply(FakeOutput(scheduled_new_reqs=[FakeNewReq("a", ([1],), 0)]))
    state.apply(
        FakeOutput(
            scheduled_cached_reqs=FakeCached(
                req_ids=["a"], new_block_ids=[None], num_computed_tokens=[9]
            )
        )
    )

    assert state.get("a").block_ids == [1]
    assert state.get("a").num_computed_tokens == 9


def test_finished_requests_are_dropped():
    state = BatchState()
    state.apply(FakeOutput(scheduled_new_reqs=[FakeNewReq("a", ([1],), 0)]))
    state.apply(FakeOutput(finished_req_ids={"a"}))

    assert "a" not in state
    assert len(state) == 0


def test_a_finished_request_id_can_be_reused_in_the_same_step():
    state = BatchState()
    state.apply(FakeOutput(scheduled_new_reqs=[FakeNewReq("a", ([1],), 0)]))

    state.apply(
        FakeOutput(
            scheduled_new_reqs=[FakeNewReq("a", ([7],), 0)],
            finished_req_ids={"a"},
        )
    )

    assert state.get("a").block_ids == [7]


def test_duplicate_active_request_id_is_rejected():
    state = BatchState()
    state.apply(FakeOutput(scheduled_new_reqs=[FakeNewReq("a", ([1],), 0)]))

    with pytest.raises(ValueError, match="already active"):
        state.apply(FakeOutput(scheduled_new_reqs=[FakeNewReq("a", ([2],), 0)]))


def test_pages_for_step_follows_the_scheduled_token_order():
    state = BatchState()
    state.apply(
        FakeOutput(
            scheduled_new_reqs=[
                FakeNewReq("a", ([1, 2],), 0),
                FakeNewReq("b", ([3],), 0),
            ]
        )
    )
    pages = state.pages_for_step(FakeOutput(num_scheduled_tokens={"b": 5, "a": 18}))

    assert [p.request_id for p in pages] == ["b", "a"]
    assert [p.num_scheduled_tokens for p in pages] == [5, 18]


def test_more_than_one_kv_cache_group_is_rejected():
    """Hybrid models produce several groups. Silently taking the first would
    serve attention from a Mamba layer's pages."""
    state = BatchState()
    with pytest.raises(ValueError, match="2 KV cache groups"):
        state.apply(
            FakeOutput(scheduled_new_reqs=[FakeNewReq("a", ([1], [2]), 0)])
        )


def test_a_cached_request_never_introduced_is_an_error():
    state = BatchState()
    with pytest.raises(KeyError, match="never introduced"):
        state.apply(
            FakeOutput(
                scheduled_cached_reqs=FakeCached(
                    req_ids=["ghost"], new_block_ids=[None], num_computed_tokens=[1]
                )
            )
        )


def test_ragged_cached_request_data_is_rejected():
    state = BatchState()
    state.apply(FakeOutput(scheduled_new_reqs=[FakeNewReq("a", ([1],), 0)]))
    with pytest.raises(ValueError, match="ragged"):
        state.apply(
            FakeOutput(
                scheduled_cached_reqs=FakeCached(
                    req_ids=["a"], new_block_ids=[], num_computed_tokens=[1]
                )
            )
        )


# --- prefill-first ordering for the attention split ----------------------


def _scheduler_output(scheduled):
    return SimpleNamespace(
        finished_req_ids=(),
        scheduled_new_reqs=(),
        scheduled_cached_reqs=None,
        num_scheduled_tokens=scheduled,
    )


def _state_with(ids):
    state = BatchState()
    for request_id in ids:
        state.apply(
            SimpleNamespace(
                finished_req_ids=(),
                scheduled_new_reqs=(
                    SimpleNamespace(
                        req_id=request_id,
                        block_ids=([1, 2, 3],),
                        num_computed_tokens=16,
                    ),
                ),
                scheduled_cached_reqs=None,
            )
        )
    return state


def test_chunk_prefilling_requests_come_first():
    # Attention slices the batch at a fixed row to send the single-token rows to
    # the decode kernel, so the two kinds have to be contiguous.
    state = _state_with(["a", "b", "c", "d"])
    pages = state.pages_for_step(
        _scheduler_output({"a": 1, "b": 40, "c": 1, "d": 7})
    )

    assert [p.request_id for p in pages] == ["b", "d", "a", "c"]
    assert [p.num_scheduled_tokens for p in pages] == [40, 7, 1, 1]


def test_ordering_is_stable_within_each_half():
    # vLLM's own order carries information -- the sampled rows are attributed by
    # it -- so only the prefill/decode partition may reorder anything.
    state = _state_with(["a", "b", "c", "d", "e"])
    pages = state.pages_for_step(
        _scheduler_output({"a": 1, "b": 5, "c": 1, "d": 5, "e": 1})
    )

    assert [p.request_id for p in pages] == ["b", "d", "a", "c", "e"]


def test_an_all_decode_step_is_left_alone():
    state = _state_with(["a", "b", "c"])
    pages = state.pages_for_step(_scheduler_output({"a": 1, "b": 1, "c": 1}))

    assert [p.request_id for p in pages] == ["a", "b", "c"]
