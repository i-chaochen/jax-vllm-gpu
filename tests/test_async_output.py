# SPDX-License-Identifier: Apache-2.0
"""Contracts for nonblocking JAX token readback."""

import numpy as np
import pytest

pytest.importorskip("vllm", reason="vLLM is supplied by the runtime image")

from jax_vllm_plugin.async_output import JaxAsyncModelRunnerOutput


class FakeTokens:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.copy_started = False

    def copy_to_host_async(self):
        self.copy_started = True

    def __array__(self, dtype=None, copy=None):
        del copy
        return np.asarray(self.values, dtype=dtype)


class FakeRunner:
    def _model_output(self, requests, sampled, states=None):
        return requests, sampled, states


def test_async_output_starts_copy_and_materializes_once(monkeypatch):
    tokens = FakeTokens([3, 4])
    completed = []
    monkeypatch.setattr("jax.device_get", lambda value: value)
    output = JaxAsyncModelRunnerOutput(
        FakeRunner(),
        ("a", "b"),
        tokens,
        states=("state-a", "state-b"),
        on_complete=lambda: completed.append(True),
    )

    assert tokens.copy_started
    requests, sampled, states = output.get_output()
    assert requests == ("a", "b")
    # Captured at submission, so a retired or reused request id cannot redirect it.
    assert states == ("state-a", "state-b")
    np.testing.assert_array_equal(sampled, np.array([3, 4], np.int64))
    assert completed == [True]

    with pytest.raises(RuntimeError, match="only be consumed once"):
        output.get_output()
    assert completed == [True]
