# SPDX-License-Identifier: Apache-2.0
"""What the runner feeds the model each step.

The runner takes its token ids, positions and sample rows from the host arrays
`prepare_step` builds. The equivalence test pins that these are the arrays it
used to reconstruct itself -- tokens concatenated in request order, the page
table's write positions, and each request's last-token index read back from
`StepView.cu_seqlens_q` -- so dropping that device-to-host copy changed nothing
the model sees.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from jax_vllm_gpu.kv.execution.bucketing import StepShapePlanner
from jax_vllm_gpu.plugin.kv_reuse import PageReuseTracker
from jax_vllm_gpu.plugin.model_runner import (
    _attention_max_seqlen_q,
    _next_rng,
    _pad,
    _prepared_model_inputs,
)
from jax_vllm_gpu.plugin.page_table import RequestPages
from jax_vllm_gpu.plugin.step import prepare_step

PAGE = 16
POOL = 64


def planner() -> StepShapePlanner:
    return StepShapePlanner(tokens_per_page=PAGE, max_batch=8, max_context_len=512, pool_pages=POOL)


def tokens_for(request: RequestPages) -> np.ndarray:
    # Distinct per request, so a misordered concatenation shows.
    base = 1000 * (ord(request.request_id) - ord("a") + 1) + request.num_computed_tokens
    return np.arange(base, base + request.num_scheduled_tokens, dtype=np.int32)


def reconstructed(requests, prepared, shape):
    """The three arrays as the runner used to build them."""
    flat = np.zeros((shape.num_tokens,), dtype=np.int32)
    cursor = 0
    for request in requests:
        tokens = tokens_for(request)
        flat[cursor : cursor + tokens.size] = tokens
        cursor += tokens.size
    positions = _pad(prepared.table.write_positions, shape.num_tokens, 0)
    rows = _pad(
        np.asarray(prepared.view.cu_seqlens_q)[1 : len(requests) + 1] - 1,
        shape.num_requests,
        0,
    )
    return flat, positions, rows


STEPS = {
    "prefill": [RequestPages("a", [1, 2], 0, 20), RequestPages("b", [3], 0, 5)],
    "mixed": [RequestPages("a", [1, 2], 16, 9), RequestPages("b", [3], 7, 1)],
    "decode": [RequestPages("a", [1, 2], 20, 1), RequestPages("b", [3], 7, 1)],
}


@pytest.mark.parametrize("step", sorted(STEPS))
def test_prepared_inputs_are_what_the_runner_used_to_reconstruct(step):
    requests = STEPS[step]
    query_lens = [r.num_scheduled_tokens for r in requests]
    max_seq_len = max(r.num_computed_tokens + r.num_scheduled_tokens for r in requests)
    if all(length == 1 for length in query_lens):
        shape = planner().decode_shape(len(requests), max_seq_len)
    else:
        shape = planner().extend_shape(
            num_tokens=sum(query_lens), max_seq_len=max_seq_len, num_requests=len(requests)
        )
    prepared = prepare_step(
        requests,
        tracker=PageReuseTracker(num_pages=POOL, tokens_per_page=PAGE),
        scrub=lambda pages: None,
        shape=shape,
        tokens_per_page=PAGE,
        token_ids=tokens_for,
    )

    got = _prepared_model_inputs(
        prepared.inputs,
        is_decode=shape.is_decode,
        active_requests=len(requests),
        request_width=shape.num_requests,
    )

    for name, actual, expected in zip(
        ("token ids", "positions", "sample rows"), got, reconstructed(requests, prepared, shape)
    ):
        np.testing.assert_array_equal(actual, expected, err_msg=name)
    assert max(query_lens) <= _attention_max_seqlen_q(shape)


def test_attention_query_bound_uses_context_bucket_not_total_mixed_tokens():
    shape = SimpleNamespace(is_decode=False, num_tokens=8192, max_seqlen_k=1024)

    assert _attention_max_seqlen_q(shape) == 1024


def test_attention_query_bound_cannot_exceed_small_token_bucket():
    shape = SimpleNamespace(is_decode=False, num_tokens=64, max_seqlen_k=2048)

    assert _attention_max_seqlen_q(shape) == 64


def test_attention_query_bound_is_one_for_decode():
    shape = SimpleNamespace(is_decode=True, num_tokens=128, max_seqlen_k=2048)

    assert _attention_max_seqlen_q(shape) == 1


def test_greedy_batch_does_not_split_rng():
    calls = []

    current, step = _next_rng("rng", all_greedy=True, split=calls.append)

    assert (current, step) == ("rng", "rng")
    assert calls == []


def test_stochastic_batch_advances_rng():
    current, step = _next_rng(
        "rng", all_greedy=False, split=lambda rng: (f"{rng}-next", f"{rng}-step")
    )

    assert (current, step) == ("rng-next", "rng-step")
