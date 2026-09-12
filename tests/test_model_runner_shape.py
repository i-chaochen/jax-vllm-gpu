# SPDX-License-Identifier: Apache-2.0
"""Shape-selection contracts for the JAX model runner."""

from types import SimpleNamespace

import numpy as np

from maxtext.inference.kv_execution.bucketing import StepShapePlanner

from jax_vllm_plugin.kv_reuse import PageReuseTracker
from jax_vllm_plugin.model_runner import (
    JaxModelRunner,
    RequestTokens,
    _attention_max_seqlen_q,
    _chained_token_sources,
    _densify_token_rungs,
    _next_rng,
    _pack_step_operands,
    _prepared_model_inputs,
    _unpack_step_operands,
)
from jax_vllm_plugin.page_table import RequestPages


def runner() -> JaxModelRunner:
    value = JaxModelRunner.__new__(JaxModelRunner)
    value.planner = StepShapePlanner(
        tokens_per_page=16,
        max_batch=8,
        max_context_len=512,
        pool_pages=256,
        max_batched_tokens=512,
    )
    value._max_request_slots = 8
    value._free_token_slots = list(range(7, -1, -1))
    value._async_scheduling = False
    return value


def test_runner_admission_rotates_page_owner_when_an_id_is_reused():
    value = runner()
    value.tracker = PageReuseTracker(num_pages=16, tokens_per_page=16)
    value.requests = {}
    admitted = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="same", prompt_token_ids=[1], sampling_params=None)
        ]
    )
    value._admit(admitted)
    first = [RequestPages("same", [3], 0, 8)]
    value.tracker.note_written(first)

    value._retire(SimpleNamespace(finished_req_ids={"same"}))
    value._admit(admitted)

    assert list(value.tracker.observe([RequestPages("same", [3], 0, 4)])) == [3]


def test_extend_shape_pins_request_axis():
    one_request = [RequestPages("a", (), 0, 128)]
    two_requests = [
        RequestPages("a", (), 64, 64),
        RequestPages("b", (), 64, 64),
    ]

    first = runner()._shape_for(one_request)
    second = runner()._shape_for(two_requests)

    assert first == second
    assert first.num_requests == 8


def test_decode_shape_still_buckets_live_request_count():
    one_request = [RequestPages("a", (), 127, 1)]
    two_requests = [
        RequestPages("a", (), 127, 1),
        RequestPages("b", (), 127, 1),
    ]

    assert runner()._shape_for(one_request).num_requests == 1
    assert runner()._shape_for(two_requests).num_requests == 2


def test_attention_query_bound_uses_context_bucket_not_total_mixed_tokens():
    shape = SimpleNamespace(
        is_decode=False,
        num_tokens=8192,
        max_seqlen_k=1024,
    )

    assert _attention_max_seqlen_q(shape) == 1024


def test_step_operands_round_trip_through_two_host_buffers():
    view = SimpleNamespace(
        slot_mapping=np.array([10, 11, -1, -1], np.int32),
        kv_indptr=np.array([0, 2, 3], np.int32),
        kv_page_indices=np.array([4, 5, 6], np.int32),
        kv_last_page_lens=np.array([8, 9], np.int32),
        cu_seqlens_q=np.array([0, 1, 2], np.int32),
    )
    original = (
        np.array([101, 102, 0, 0], np.int32),
        np.array([20, 30, 0, 0], np.int32),
        view.slot_mapping,
        view.kv_indptr,
        view.kv_page_indices,
        view.kv_last_page_lens,
        view.cu_seqlens_q,
        np.array([0, 1], np.int32),
        np.array([0.0, 0.5], np.float32),
        np.array([1.0, 0.9], np.float32),
        np.array([1, 8], np.int32),
    )

    packed = _pack_step_operands(
        original[0],
        original[1],
        view,
        original[7],
        original[8],
        original[9],
        original[10],
    )
    restored = _unpack_step_operands(
        *packed, num_tokens=4, num_requests=2, num_pages=3
    )

    for actual, expected in zip(restored, original, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_async_step_operands_include_device_token_chaining_metadata():
    view = SimpleNamespace(
        slot_mapping=np.array([10, 11], np.int32),
        kv_indptr=np.array([0, 1, 2], np.int32),
        kv_page_indices=np.array([4, 5], np.int32),
        kv_last_page_lens=np.array([8, 9], np.int32),
        cu_seqlens_q=np.array([0, 1, 2], np.int32),
    )
    chain_sources = np.array([-1, 7], np.int32)
    request_slots = np.array([3, 7], np.int32)
    packed = _pack_step_operands(
        np.array([101, 0], np.int32),
        np.array([20, 21], np.int32),
        view,
        np.array([0, 1], np.int32),
        np.array([0.0, 0.0], np.float32),
        np.array([1.0, 1.0], np.float32),
        np.array([1, 1], np.int32),
        chained_token_sources=chain_sources,
        request_token_slots=request_slots,
    )

    restored = _unpack_step_operands(
        *packed,
        num_tokens=2,
        num_requests=2,
        num_pages=2,
        include_chaining=True,
    )

    np.testing.assert_array_equal(restored[-2], chain_sources)
    np.testing.assert_array_equal(restored[-1], request_slots)


def test_asm_dense_page_map_survives_the_packed_device_transfer():
    """The host-built map must reach the PagedPlan reconstructed inside jit.

    The first serving attempt built `StepView.block_tables` correctly but
    `_pack_step_operands` silently dropped it, so ASM failed only after the
    server admitted all 512 requests.
    """
    block_tables = np.array([[4, 5, 0], [6, 7, 8]], np.int32)
    context_lens = np.array([24, 48], np.int32)
    view = SimpleNamespace(
        slot_mapping=np.array([10, 11], np.int32),
        kv_indptr=np.array([0, 2, 5], np.int32),
        kv_page_indices=np.array([4, 5, 6, 7, 8], np.int32),
        kv_last_page_lens=np.array([8, 16], np.int32),
        cu_seqlens_q=np.array([0, 1, 2], np.int32),
        block_tables=block_tables,
        seq_lens=context_lens,
    )

    packed = _pack_step_operands(
        np.array([101, 102], np.int32),
        np.array([20, 47], np.int32),
        view,
        np.array([0, 1], np.int32),
        np.array([0.0, 0.0], np.float32),
        np.array([1.0, 1.0], np.float32),
        np.array([1, 1], np.int32),
        chained_token_sources=np.array([-1, 7], np.int32),
        request_token_slots=np.array([3, 7], np.int32),
    )
    restored = _unpack_step_operands(
        *packed,
        num_tokens=2,
        num_requests=2,
        num_pages=5,
        block_table_blocks=3,
        include_chaining=True,
    )

    np.testing.assert_array_equal(restored[11], block_tables)
    np.testing.assert_array_equal(restored[12], context_lens)
    np.testing.assert_array_equal(restored[-2], np.array([-1, 7], np.int32))
    np.testing.assert_array_equal(restored[-1], np.array([3, 7], np.int32))


def test_async_decode_uses_pending_device_token_for_missing_host_token():
    request = RequestPages("request", (), 3, 1)
    state = RequestTokens("request", [10, 11, 12], None, device_token_slot=7)
    value = runner()
    value._async_scheduling = True
    value.requests = {"request": state}

    np.testing.assert_array_equal(value._tokens_for(request), np.array([0], np.int32))
    np.testing.assert_array_equal(
        _chained_token_sources(
            [request],
            value.requests,
            num_tokens=4,
            sample_rows=np.arange(4, dtype=np.int32),
        ),
        np.array([7, -1, -1, -1], np.int32),
    )


def test_async_extend_maps_trailing_pending_token_to_flattened_row():
    requests = [
        RequestPages("a", (), 2, 3),
        RequestPages("b", (), 0, 2),
    ]
    states = {
        "a": RequestTokens("a", [1, 2, 3, 4], None, device_token_slot=5),
        "b": RequestTokens("b", [8, 9], None, device_token_slot=6),
    }

    np.testing.assert_array_equal(
        _chained_token_sources(
            requests,
            states,
            num_tokens=8,
            # Contiguous extend packing: "a" holds tokens [0, 3) and "b" [3, 5),
            # so their last tokens -- what `sample_at` records -- are 2 and 4.
            sample_rows=np.array([2, 4], np.int32),
        ),
        np.array([-1, -1, 5, -1, -1, -1, -1, -1], np.int32),
    )


def test_async_split_step_chains_pending_token_to_the_split_decode_offset():
    """A split mixed step packs decode tokens after the padded prefill capacity.

    The regression this guards: deriving the target by walking
    `num_scheduled_tokens` puts the pending token at the end of the *true*
    prefill tokens, which under the split is an inert `slot_mapping=-1` slot,
    while the real decode position keeps its zero placeholder.
    """
    requests = [
        # One chunk-prefilling request occupying tokens [0, 3)...
        RequestPages("p", (), 0, 3),
        # ...then two decode rows, which the split packs at capacity 8 and 9,
        # not at the contiguous cursor 3 and 4.
        RequestPages("d0", (), 5, 1),
        RequestPages("d1", (), 7, 1),
    ]
    states = {
        "p": RequestTokens("p", [1, 2, 3], None, device_token_slot=1),
        "d0": RequestTokens("d0", [1, 2, 3, 4, 5], None, device_token_slot=4),
        "d1": RequestTokens("d1", [1, 2, 3, 4, 5, 6, 7], None, device_token_slot=6),
    }
    sample_rows = np.array([2, 8, 9], np.int32)

    sources = _chained_token_sources(
        requests, states, num_tokens=10, sample_rows=sample_rows
    )

    expected = np.full((10,), -1, np.int32)
    expected[8] = 4
    expected[9] = 6
    np.testing.assert_array_equal(sources, expected)
    # The contiguous positions the old cursor would have written must stay clear.
    assert sources[3] == -1
    assert sources[4] == -1


def test_attention_query_bound_cannot_exceed_small_token_bucket():
    shape = SimpleNamespace(
        is_decode=False,
        num_tokens=64,
        max_seqlen_k=2048,
    )

    assert _attention_max_seqlen_q(shape) == 64


def test_attention_query_bound_is_one_for_decode():
    shape = SimpleNamespace(
        is_decode=True,
        num_tokens=128,
        max_seqlen_k=2048,
    )

    assert _attention_max_seqlen_q(shape) == 1


def test_token_rungs_leave_largest_interval_coarse():
    assert _densify_token_rungs((64, 128, 256, 512)) == (
        64,
        96,
        128,
        192,
        256,
        512,
    )


def test_token_rungs_support_quarter_intervals():
    assert _densify_token_rungs((64, 128, 256, 512), subdivisions=4) == (
        64,
        80,
        96,
        112,
        128,
        160,
        192,
        224,
        256,
        512,
    )


def test_token_rungs_reject_nonpositive_subdivisions():
    with np.testing.assert_raises_regex(ValueError, "must be positive"):
        _densify_token_rungs((64, 128), subdivisions=0)


def test_single_token_rung_is_unchanged():
    assert _densify_token_rungs((64,)) == (64,)


def test_greedy_batch_does_not_split_rng():
    calls = []

    current, step = _next_rng(
        "rng",
        all_greedy=True,
        split=lambda rng: calls.append(rng),
    )

    assert (current, step) == ("rng", "rng")
    assert calls == []


def test_stochastic_batch_advances_rng():
    current, step = _next_rng(
        "rng",
        all_greedy=False,
        split=lambda rng: (f"{rng}-next", f"{rng}-step"),
    )

    assert (current, step) == ("rng-next", "rng-step")


def test_prepared_extend_inputs_are_reused_without_reconstruction():
    inputs = SimpleNamespace(
        tokens=np.asarray([[11, 12, 13, 0]], dtype=np.int32),
        positions=np.asarray([[4, 5, 9, 0]], dtype=np.int32),
        sample_rows=np.asarray([0, 0], dtype=np.int32),
        sample_at=np.asarray([1, 2], dtype=np.int32),
    )

    token_ids, positions, sample_rows = _prepared_model_inputs(
        inputs,
        is_decode=False,
        active_requests=2,
        request_width=4,
    )

    np.testing.assert_array_equal(token_ids, [11, 12, 13, 0])
    np.testing.assert_array_equal(positions, [4, 5, 9, 0])
    np.testing.assert_array_equal(sample_rows, [1, 2, 0, 0])


def test_prepared_decode_inputs_keep_only_active_sample_rows():
    inputs = SimpleNamespace(
        tokens=np.asarray([[21], [22], [0], [0]], dtype=np.int32),
        positions=np.asarray([[7], [3], [0], [0]], dtype=np.int32),
        sample_rows=np.arange(4, dtype=np.int32),
        sample_at=np.zeros((4,), dtype=np.int32),
    )

    token_ids, positions, sample_rows = _prepared_model_inputs(
        inputs,
        is_decode=True,
        active_requests=2,
        request_width=4,
    )

    np.testing.assert_array_equal(token_ids, [21, 22, 0, 0])
    np.testing.assert_array_equal(positions, [7, 3, 0, 0])
    np.testing.assert_array_equal(sample_rows, [0, 1, 0, 0])


def _ledger_runner(*, async_scheduling: bool) -> JaxModelRunner:
    """A runner carrying just enough state to emit one step ledger record."""
    value = runner()
    value._async_scheduling = async_scheduling
    value._step_telemetry = True
    value._step_telemetry_index = 0
    value.requests = {"a": SimpleNamespace(num_prompt_tokens=100)}
    value.batch_state = {"a": None}
    return value


def _ledger_call(value: JaxModelRunner) -> None:
    prepared = SimpleNamespace(
        view=SimpleNamespace(num_active_requests=3, num_active_tokens=40),
        table=SimpleNamespace(
            query_lens=[38, 1, 1],
            seq_lens=[300, 120, 90],
            page_ids=[range(19), range(8), range(6)],
        ),
    )
    shape = SimpleNamespace(
        num_requests=8,
        num_tokens=64,
        num_pages=256,
        max_seqlen_k=512,
        is_decode=False,
    )
    value._emit_step_ledger(
        [SimpleNamespace(request_id="a", num_computed_tokens=0, num_scheduled_tokens=40)],
        prepared=prepared,
        shape=shape,
        max_seqlen_q=64,
        step_start_ns=1_000,
        execute_start_ns=3_000,
        execute_end_ns=9_000,
    )


def _logged_record(caplog) -> dict:
    import json

    for message in caplog.text.splitlines():
        if "JAX_VLLM_STEP " in message:
            return json.loads(message.split("JAX_VLLM_STEP ", 1)[1])
    raise AssertionError(f"no step ledger record logged: {caplog.text!r}")


def test_step_ledger_records_padding_on_every_axis(caplog):
    value = _ledger_runner(async_scheduling=False)

    with caplog.at_level("WARNING", logger="jax_vllm_plugin.model_runner"):
        _ledger_call(value)

    record = _logged_record(caplog)
    # The padding this measures is the point of the ledger: each axis is
    # rounded up independently, so each has to be recoverable independently.
    assert record["true_shape"] == {
        "num_requests": 3,
        "num_tokens": 40,
        "num_pages": 33,
        "max_seqlen_q": 38,
        "max_seqlen_k": 300,
    }
    assert record["padded_shape"]["num_requests"] == 8
    assert record["padded_shape"]["num_tokens"] == 64
    assert record["padded_shape"]["num_pages"] == 256
    assert record["padded_shape"]["max_seqlen_k"] == 512


def test_step_ledger_under_async_reports_submit_not_blocking_execution(caplog):
    value = _ledger_runner(async_scheduling=True)

    with caplog.at_level("WARNING", logger="jax_vllm_plugin.model_runner"):
        _ledger_call(value)

    record = _logged_record(caplog)
    assert record["async_scheduling"] is True
    # Asynchronously the second interval covers submission only. Reporting it
    # as `execute_blocking` would silently understate device time.
    assert record["timing_ns"]["submit"] == 6_000
    assert "execute_blocking" not in record["timing_ns"]
    assert record["timing_ns"]["prepare"] == 2_000


def test_step_ledger_under_sync_still_reports_blocking_execution(caplog):
    value = _ledger_runner(async_scheduling=False)

    with caplog.at_level("WARNING", logger="jax_vllm_plugin.model_runner"):
        _ledger_call(value)

    record = _logged_record(caplog)
    assert record["async_scheduling"] is False
    assert record["timing_ns"]["execute_blocking"] == 6_000
    assert "submit" not in record["timing_ns"]


# --- the prefill/decode attention split ----------------------------------

from jax_vllm_plugin.model_runner import PREFILL_SLOTS, count_prefill  # noqa: E402

# Production geometry: 128-request concurrency and an 8,192-token budget, which
# is what makes the eight reserved slots and the 120-row decode half meaningful.
PRODUCTION_PLANNER = StepShapePlanner(
    tokens_per_page=16,
    max_batch=128,
    max_context_len=2048,
    pool_pages=64591,
    max_batched_tokens=8192,
)


class _Req:
    def __init__(self, num_scheduled_tokens):
        self.num_scheduled_tokens = num_scheduled_tokens


def test_an_all_prefill_batch_is_not_split(planner=PRODUCTION_PLANNER):
    # No decode rows means the split would add a decode kernel over rows holding
    # no request, so the batch keeps the single-kernel shape.
    shape = planner.extend_shape(
        128, 610, num_prefill_requests=1, num_prefill_tokens=128,
        prefill_slots=PREFILL_SLOTS,
    )
    assert shape.num_prefill_slots is None


def test_count_prefill_counts_only_the_leading_run():
    assert count_prefill([_Req(40), _Req(7), _Req(1), _Req(1)]) == 2
    assert count_prefill([_Req(1), _Req(1)]) == 0
    assert count_prefill([_Req(9)]) == 1


def test_count_prefill_stops_at_the_first_decode_row():
    # An unsorted batch must not be counted past: the split hands the prefill
    # kernel rows [0, slots), and a multi-token request sitting after a
    # single-token one would have its tokens in the decode half.
    assert count_prefill([_Req(40), _Req(1), _Req(40)]) == 1


def test_a_mixed_step_within_the_reservation_selects_the_split_shape(planner=PRODUCTION_PLANNER):
    # 2 prefilling requests carrying 708 tokens, beside 125 decoding ones.
    shape = planner.extend_shape(
        833, 610, num_prefill_requests=2, num_prefill_tokens=708,
        prefill_slots=PREFILL_SLOTS,
    )

    assert shape.num_prefill_slots == PREFILL_SLOTS
    # The decode half is one token per remaining row, so the prefill capacity
    # follows from the token bucket rather than adding a ladder axis.
    decode_rows = shape.num_requests - PREFILL_SLOTS
    assert shape.num_prefill_tokens == shape.num_tokens - decode_rows


def test_more_prefilling_requests_than_slots_falls_back_to_one_kernel(planner=PRODUCTION_PLANNER):
    shape = planner.extend_shape(
        8192, 610, num_prefill_requests=21, num_prefill_tokens=8100,
        prefill_slots=PREFILL_SLOTS,
    )

    assert shape.num_prefill_slots is None
    assert shape == planner.extend_shape(8192, 610)


def test_an_all_decode_batch_never_takes_the_split_shape(planner=PRODUCTION_PLANNER):
    shape = planner.extend_shape(
        128, 610, num_prefill_requests=0, num_prefill_tokens=0,
        prefill_slots=PREFILL_SLOTS,
    )
    assert shape.num_prefill_slots is None


def test_the_split_adds_no_new_shape_axis(planner=PRODUCTION_PLANNER):
    # The reservation is a fixed boundary, not a ladder, so sweeping the token
    # count must not multiply the shape family.
    tokens = range(200, 8100, 37)
    split = {
        planner.extend_shape(
            t, 610, num_prefill_requests=2, num_prefill_tokens=t - 125,
            prefill_slots=PREFILL_SLOTS,
        )
        for t in tokens
    }
    unsplit = {planner.extend_shape(t, 610) for t in tokens}
    assert len(split) <= len(unsplit) + 1
