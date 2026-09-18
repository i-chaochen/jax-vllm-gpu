# SPDX-License-Identifier: Apache-2.0
"""Nonblocking JAX token readback for vLLM's asynchronous scheduler."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from vllm.v1.outputs import AsyncModelRunnerOutput


class JaxAsyncModelRunnerOutput(AsyncModelRunnerOutput):
    """Keep device tokens alive while vLLM schedules the next model step."""

    def __init__(
        self,
        runner: Any,
        requests: tuple[Any, ...],
        tokens: Any,
        *,
        states: tuple[Any, ...] | None = None,
        on_complete: Callable[[], None] | None = None,
        step: Any = None,
    ) -> None:
        self._runner = runner
        self._requests = requests
        # The per-request token state captured at submission. Resolving it later
        # through `runner.requests` could hit a retired or reused request id.
        self._states = states
        self._tokens = tokens
        self._on_complete = on_complete
        # The PendingStep this output finishes, carried so the step's duration is
        # recorded against when it was dispatched rather than when it was read.
        self._step = step
        self._consumed = False

        copy_to_host_async = getattr(tokens, "copy_to_host_async", None)
        if copy_to_host_async is not None:
            copy_to_host_async()

    def get_output(self) -> Any:
        if self._consumed:
            raise RuntimeError("an async model output can only be consumed once")
        self._consumed = True

        import jax

        try:
            sampled = np.asarray(jax.device_get(self._tokens), dtype=np.int64)
            return self._runner._model_output(
                self._requests, sampled, states=self._states, step=self._step
            )
        finally:
            if self._on_complete is not None:
                self._on_complete()
