# SPDX-License-Identifier: Apache-2.0
"""Skips `gpu_only` tests when JAX sees no GPU.

Unmarked tests run everywhere. Nothing here skips them on a GPU host, so a
run's pass and skip counts say what it actually exercised.
"""

from __future__ import annotations

import functools

import pytest


@functools.cache
def _jax_sees_a_gpu() -> bool:
    import jax

    return jax.default_backend() != "cpu"


def pytest_collection_modifyitems(config, items):
    gpu_items = [item for item in items if item.get_closest_marker("gpu_only")]
    if not gpu_items or _jax_sees_a_gpu():
        return
    skip = pytest.mark.skip(reason="needs a GPU, and JAX sees none")
    for item in gpu_items:
        item.add_marker(skip)
