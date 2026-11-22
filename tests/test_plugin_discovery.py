# SPDX-License-Identifier: Apache-2.0
"""Discovery tests: the four ways this plugin can fail to be picked up.

All four present identically from the outside -- vLLM quietly runs on some
other platform -- so each needs its own assertion rather than a manual check.
"""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.metadata import entry_points

import pytest

from vllm_xla_gpu.plugin.platform import (
    FORCE_ENV_VAR,
    PLATFORM_QUALNAME,
    JaxPlatform,
    register,
)

vllm = pytest.importorskip("vllm", reason="vLLM is supplied by the runtime image")


def test_the_entry_points_are_declared_in_installed_metadata():
    """Catches a misspelled group or a wheel built without the entry points."""
    platform_group = entry_points(group="vllm.platform_plugins")
    assert "vllm_xla_gpu" in {ep.name for ep in platform_group}

    general_group = entry_points(group="vllm.general_plugins")
    assert "vllm_xla_gpu_kv_connector" in {ep.name for ep in general_group}


def test_the_probe_returns_a_qualname_string_rather_than_the_class():
    """The error this catches is silent: vLLM calls the entry point and then
    resolves the return value as a qualname, so returning the Platform class
    yields an instance where a string is required."""
    from vllm.utils.import_utils import resolve_obj_by_qualname

    os.environ[FORCE_ENV_VAR] = "1"
    try:
        returned = register()
    finally:
        del os.environ[FORCE_ENV_VAR]

    assert isinstance(returned, str)
    assert resolve_obj_by_qualname(returned) is JaxPlatform


def test_the_probe_declines_when_explicitly_disabled():
    os.environ[FORCE_ENV_VAR] = "0"
    try:
        assert register() is None
    finally:
        del os.environ[FORCE_ENV_VAR]


# Resolution is cached in vllm.platforms._current_platform on first access, so
# these run in a fresh interpreter. That also matches how it happens in
# production, where resolution occurs once per process.
_RESOLVE_SCRIPT = """
import vllm.platforms
print(type(vllm.platforms.current_platform).__module__
      + "."
      + type(vllm.platforms.current_platform).__name__)
"""


def _resolve_platform_in_subprocess(env: dict[str, str]) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _RESOLVE_SCRIPT],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().splitlines()[-1]


def test_this_plugin_wins_over_the_builtin_rocm_platform():
    """The whole discovery chain, and the reason no vLLM patch is needed: an
    out-of-tree plugin is preferred over every built-in, so on a ROCm host this
    beats vllm.platforms.rocm.RocmPlatform without an upstream change."""
    resolved = _resolve_platform_in_subprocess({FORCE_ENV_VAR: "1"})
    assert resolved == PLATFORM_QUALNAME


def test_an_explicit_cpu_target_silently_bypasses_plugin_loading():
    """Guards a real deployment hazard rather than our own code. vLLM returns
    the CPU platform before plugins are loaded at all when VLLM_TARGET_DEVICE
    is cpu, so if that leaks in from a base image the failure looks exactly
    like the wheel being absent. Asserted here so the behaviour is known
    rather than discovered."""
    resolved = _resolve_platform_in_subprocess(
        {FORCE_ENV_VAR: "1", "VLLM_TARGET_DEVICE": "cpu"}
    )
    assert resolved != PLATFORM_QUALNAME
