# SPDX-License-Identifier: Apache-2.0
"""Optional ROCTX ranges, for lining up serving steps with GPU work in rocprofv3."""

from __future__ import annotations

import ctypes
import ctypes.util
import os

ROCTX_ENV_VAR = "JAX_VLLM_ROCTX"


class Roctx:
    """Thin binding to rocprofiler-sdk-roctx.

    ROCm also ships a legacy `libroctx64` compatibility library, but rocprofv3
    does not record ranges emitted through it, so this binds the SDK library,
    which initializes the profiler's marker domain.
    """

    def __init__(self, library) -> None:
        self._push = library.roctxRangePushA
        self._push.argtypes = [ctypes.c_char_p]
        self._push.restype = ctypes.c_int
        self._pop = library.roctxRangePop
        self._pop.argtypes = []
        self._pop.restype = ctypes.c_int
        self._start = library.roctxRangeStartA
        self._start.argtypes = [ctypes.c_char_p]
        self._start.restype = ctypes.c_uint64
        self._stop = library.roctxRangeStop
        self._stop.argtypes = [ctypes.c_uint64]
        self._stop.restype = None

    @classmethod
    def from_environment(cls) -> "Roctx | None":
        """The binding when `JAX_VLLM_ROCTX=1`, else None without loading anything."""
        if os.environ.get(ROCTX_ENV_VAR) != "1":
            return None
        path = ctypes.util.find_library("rocprofiler-sdk-roctx")
        if path is None:
            raise RuntimeError(f"{ROCTX_ENV_VAR}=1 but librocprofiler-sdk-roctx is unavailable")
        return cls(ctypes.CDLL(path))

    def push(self, name: str) -> None:
        # Push and pop return the nesting level, which is nonzero for any nested
        # range; only a negative value is an error.
        if self._push(_encode(name)) < 0:
            raise RuntimeError("roctxRangePushA failed")

    def pop(self) -> None:
        if self._pop() < 0:
            raise RuntimeError("roctxRangePop failed")

    def start(self, name: str) -> int:
        """Start a range that may be stopped later, or from another host thread."""
        return int(self._start(_encode(name)))

    def stop(self, range_id: int) -> None:
        self._stop(range_id)


def _encode(name: str) -> bytes:
    if "\x00" in name:
        raise ValueError("ROCTX range names cannot contain NUL bytes")
    return name.encode("utf-8")
