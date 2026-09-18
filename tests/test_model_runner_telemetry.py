# SPDX-License-Identifier: Apache-2.0
"""Pure host-side tests for model-step telemetry semantics."""

from types import SimpleNamespace

from jax_vllm_plugin.model_runner import _run_identity, _scheduled_token_split
from jax_vllm_plugin.page_table import RequestPages


def states(**prompt_lengths):
    return {
        request_id: SimpleNamespace(num_prompt_tokens=length)
        for request_id, length in prompt_lengths.items()
    }


def test_pure_prefill_and_decode_are_split_at_prompt_boundary():
    prefill = [RequestPages("a", (), 0, 20)]
    decode = [RequestPages("a", (), 100, 1)]

    assert _scheduled_token_split(prefill, states(a=100)) == {
        "batch_kind": "prefill",
        "prefill_requests": 1,
        "decode_requests": 0,
        "scheduled_prompt_tokens": 20,
        "scheduled_output_tokens": 0,
        "scheduled_tokens": 20,
    }
    assert _scheduled_token_split(decode, states(a=100)) == {
        "batch_kind": "decode",
        "prefill_requests": 0,
        "decode_requests": 1,
        "scheduled_prompt_tokens": 0,
        "scheduled_output_tokens": 1,
        "scheduled_tokens": 1,
    }


def test_mixed_batch_counts_each_kind():
    requests = [
        RequestPages("prefill", (), 0, 20),
        RequestPages("decode", (), 100, 1),
    ]

    split = _scheduled_token_split(
        requests,
        states(prefill=100, decode=100),
    )

    assert split["batch_kind"] == "mixed"
    assert split["prefill_requests"] == 1
    assert split["decode_requests"] == 1
    assert split["scheduled_prompt_tokens"] == 20
    assert split["scheduled_output_tokens"] == 1
    assert split["scheduled_tokens"] == 21


def test_one_request_can_cross_the_prompt_boundary():
    split = _scheduled_token_split(
        [RequestPages("a", (), 8, 4)],
        states(a=10),
    )

    assert split["batch_kind"] == "mixed"
    assert split["scheduled_prompt_tokens"] == 2
    assert split["scheduled_output_tokens"] == 2


def test_run_identity_is_conservative():
    warmup = [SimpleNamespace(request_id="cmpl-warmup2-17-0")]
    measured = [SimpleNamespace(request_id="chatcmpl-measure3-4-0")]
    mixed = warmup + measured

    assert _run_identity(warmup) == ("warmup", 2)
    assert _run_identity(measured) == ("measured", 3)
    assert _run_identity([SimpleNamespace(request_id="other")]) == ("unknown", None)
    assert _run_identity(mixed) == ("cross_run", None)
