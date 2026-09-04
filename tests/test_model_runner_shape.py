# SPDX-License-Identifier: Apache-2.0
"""Shape-selection contracts for the JAX model runner."""

from types import SimpleNamespace

import numpy as np

from maxtext.inference.kv_execution.bucketing import StepShapePlanner

from jax_vllm_plugin.model_runner import (
    JaxModelRunner,
    _attention_max_seqlen_q,
    _densify_token_rungs,
    _next_rng,
    _prepared_model_inputs,
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
    return value


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
