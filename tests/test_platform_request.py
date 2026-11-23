# SPDX-License-Identifier: Apache-2.0
"""Per-request semantics accepted by the JAX sampler."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

vllm = pytest.importorskip("vllm", reason="vLLM is supplied by the runtime image")

from vllm_xla_gpu.plugin.platform import JaxPlatform


def test_temperature_top_p_and_top_k_are_supported():
    params = vllm.SamplingParams(temperature=0.7, top_p=0.9, top_k=20)

    JaxPlatform.validate_request({}, params)


def test_greedy_defaults_are_supported():
    """The defaults are what every plain request carries, so a field whose real
    default differs from the one assumed here would reject all traffic."""
    JaxPlatform.validate_request({}, vllm.SamplingParams(temperature=0.0, max_tokens=8))


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"seed": 7}, "seed"),
        ({"presence_penalty": 0.5}, "presence_penalty"),
        ({"min_p": 0.1}, "min_p"),
        ({"logprobs": 1}, "logprobs"),
        ({"min_tokens": 2}, "min_tokens"),
        ({"allowed_token_ids": [1, 2]}, "allowed_token_ids"),
    ],
)
def test_ignored_sampling_semantics_are_rejected(kwargs, field):
    params = vllm.SamplingParams(**kwargs)

    with pytest.raises(ValueError, match=field):
        JaxPlatform.validate_request({}, params)


def test_non_generation_requests_are_rejected():
    with pytest.raises(ValueError, match="generation requests only"):
        JaxPlatform.validate_request({}, SimpleNamespace())
