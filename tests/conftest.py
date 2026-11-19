# SPDX-License-Identifier: Apache-2.0
"""Skips `gpu_only` tests when JAX sees no GPU, unless the run requires one.

Unmarked tests run everywhere. Nothing here skips them on a GPU host, so a
run's pass and skip counts say what it actually exercised. On a GPU runner set
`JAX_VLLM_REQUIRE_GPU=1`: a JAX that fell back to the CPU then fails the run
rather than skipping every GPU test and reporting green.
"""

from __future__ import annotations

import functools
import os
import sys

import pytest

REQUIRE_GPU_ENV_VAR = "JAX_VLLM_REQUIRE_GPU"


@functools.cache
def _jax_sees_a_gpu() -> bool:
    import jax

    return jax.default_backend() != "cpu"


def pytest_collection_modifyitems(config, items):
    gpu_items = [item for item in items if item.get_closest_marker("gpu_only")]
    if not gpu_items or _jax_sees_a_gpu():
        return
    if os.environ.get(REQUIRE_GPU_ENV_VAR) == "1":
        raise pytest.UsageError(
            f"{REQUIRE_GPU_ENV_VAR}=1, but JAX sees no GPU, so {len(gpu_items)} gpu_only "
            "tests would be skipped"
        )
    skip = pytest.mark.skip(reason="needs a GPU, and JAX sees none")
    for item in gpu_items:
        item.add_marker(skip)


@pytest.fixture(autouse=True, scope="module")
def _clear_jax_caches():
    """Drop each module's compiled programs before the next module runs.

    Engines and runners are static arguments of their jitted steps, so the jit
    cache keeps every one of them, and its KV pool, alive until it is cleared.
    """
    yield
    jax = sys.modules.get("jax")
    if jax is not None:
        jax.clear_caches()
