# SPDX-License-Identifier: Apache-2.0
"""ROCTX profiling is optional and fails closed when requested."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jax_vllm_plugin.profiling import ROCTX_ENV_VAR, Roctx


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
        roctxProfilerPause=FakeFunction(0),
        roctxProfilerResume=FakeFunction(0),
    )


def test_disabled_profiling_does_not_load_a_library(monkeypatch):
    monkeypatch.delenv(ROCTX_ENV_VAR, raising=False)

    assert Roctx.from_environment() is None


def test_range_names_are_encoded_and_balanced():
    library = fake_library()
    roctx = Roctx(library)

    roctx.push("model call")
    roctx.pop()
    range_id = roctx.start("async model call")
    roctx.stop(range_id)
    roctx.profiler_resume()
    roctx.profiler_pause()

    assert library.roctxRangePushA.calls == [(b"model call",)]
    assert library.roctxRangePop.calls == [()]
    assert range_id == 17
    assert library.roctxRangeStartA.calls == [(b"async model call",)]
    assert library.roctxRangeStop.calls == [(17,)]
    assert library.roctxProfilerResume.calls == [(0,)]
    assert library.roctxProfilerPause.calls == [(0,)]


def test_invalid_or_failed_ranges_are_reported():
    roctx = Roctx(fake_library(push_result=1))

    with pytest.raises(ValueError, match="NUL"):
        roctx.push("bad\x00name")
    with pytest.raises(RuntimeError, match="Push"):
        roctx.push("failed")


@pytest.mark.parametrize(
    ("name", "method"),
    (("roctxProfilerResume", "profiler_resume"), ("roctxProfilerPause", "profiler_pause")),
)
def test_failed_profiler_control_is_reported(name, method):
    library = fake_library()
    getattr(library, name).result = 1
    roctx = Roctx(library)

    with pytest.raises(RuntimeError, match=name):
        getattr(roctx, method)()
