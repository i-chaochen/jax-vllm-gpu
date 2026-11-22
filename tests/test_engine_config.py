# SPDX-License-Identifier: Apache-2.0
"""The paged engine's settings, checked against the model config they serve."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jax_vllm_gpu.engine.config import PagedConfig


def a_model(**overrides):
    """The three MaxText settings `PagedConfig.resolve` reads."""
    fields = {"attention": "gpu_paged", "scan_layers": False, "max_target_length": 128}
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_the_context_length_defaults_to_the_target_length():
    """max_target_length already states the longest sequence to be served."""
    assert PagedConfig(num_blocks=256).resolve(a_model()).max_context_len == 128


def test_an_explicit_context_length_is_kept():
    assert PagedConfig(num_blocks=256, max_context_len=64).resolve(a_model()).max_context_len == 64


def test_scan_layers_is_refused():
    """Scanning stacks the caches, which would copy the whole pool every step."""
    with pytest.raises(ValueError, match="scan_layers=False"):
        PagedConfig(num_blocks=256).resolve(a_model(scan_layers=True))


def test_a_pool_size_is_required():
    """Capacity has no safe default: too small caps concurrency silently."""
    with pytest.raises(ValueError, match="num_blocks must be set"):
        PagedConfig().resolve(a_model())


def test_a_pool_too_small_for_one_request_is_refused():
    """Otherwise this surfaces as backpressure that never clears."""
    with pytest.raises(ValueError, match="no request could ever finish"):
        PagedConfig(num_blocks=2, max_context_len=1024).resolve(a_model())


def test_another_attention_kernel_is_refused():
    """The pool is threaded through `gpu_paged` attention and nothing else."""
    with pytest.raises(ValueError, match="requires attention='gpu_paged'"):
        PagedConfig(num_blocks=256).resolve(a_model(attention="dot_product"))
