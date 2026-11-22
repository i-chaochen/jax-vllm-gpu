# SPDX-License-Identifier: Apache-2.0
"""ROCTX profiling is optional, and fails closed when requested."""

from __future__ import annotations

import ctypes.util
from types import SimpleNamespace

import pytest

from vllm_xla_gpu.plugin.profiling import ROCTX_ENV_VAR, Roctx


class FakeFunction:
    def __init__(self, result=0):
        self.result = result
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.result


def fake_library(push_result=0, pop_result=0):
    return SimpleNamespace(
        roctxRangePushA=FakeFunction(push_result),
        roctxRangePop=FakeFunction(pop_result),
        roctxRangeStartA=FakeFunction(17),
        roctxRangeStop=FakeFunction(None),
    )


def test_disabled_profiling_does_not_load_a_library(monkeypatch):
    monkeypatch.delenv(ROCTX_ENV_VAR, raising=False)

    assert Roctx.from_environment() is None


def test_requested_profiling_without_the_library_fails_at_startup(monkeypatch):
    monkeypatch.setenv(ROCTX_ENV_VAR, "1")
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: None)

    with pytest.raises(RuntimeError, match="unavailable"):
        Roctx.from_environment()


def test_range_names_are_encoded_and_balanced():
    library = fake_library()
    roctx = Roctx(library)

    roctx.push("model call")
    roctx.pop()
    range_id = roctx.start("step")
    roctx.stop(range_id)

    assert library.roctxRangePushA.calls == [(b"model call",)]
    assert library.roctxRangePop.calls == [()]
    assert range_id == 17
    assert library.roctxRangeStartA.calls == [(b"step",)]
    assert library.roctxRangeStop.calls == [(17,)]


def test_a_nested_range_is_not_an_error():
    """Push and pop return the nesting level, which is 1 for a range inside
    another; reading any nonzero value as failure rejects all nesting."""
    roctx = Roctx(fake_library(push_result=1, pop_result=1))

    roctx.push("inner")
    roctx.pop()


def test_failed_and_invalid_ranges_are_reported():
    roctx = Roctx(fake_library(push_result=-1, pop_result=-1))

    with pytest.raises(RuntimeError, match="Push"):
        roctx.push("failed")
    with pytest.raises(RuntimeError, match="Pop"):
        roctx.pop()
    with pytest.raises(ValueError, match="NUL"):
        roctx.push("bad\x00name")
    with pytest.raises(ValueError, match="NUL"):
        roctx.start("bad\x00name")
