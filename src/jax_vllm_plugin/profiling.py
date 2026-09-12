# SPDX-License-Identifier: Apache-2.0
"""Optional ROCTX ranges for correlating model calls with GPU dispatches."""

from __future__ import annotations

import ctypes
import ctypes.util
import os

ROCTX_ENV_VAR = "JAX_VLLM_ROCTX"
ROCTX_SELECTED_REGIONS_ENV_VAR = "JAX_VLLM_ROCTX_SELECTED_REGIONS"
ROCTX_EXPECTED_TOKENS_ENV_VAR = "JAX_VLLM_ROCTX_EXPECTED_TOKENS"
ROCTX_PROFILE_REP_ENV_VAR = "JAX_VLLM_ROCTX_PROFILE_REP"


class Roctx:
    """Thin binding to rocprofiler-sdk-roctx.

    ROCm also ships a legacy ``libroctx64`` compatibility library. rocprofv3
    1.3 does not record ranges emitted through that library, so use the SDK
    implementation that initializes the profiler's MARKER domain.
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
        self._pause = library.roctxProfilerPause
        self._pause.argtypes = [ctypes.c_uint64]
        self._pause.restype = ctypes.c_int
        self._resume = library.roctxProfilerResume
        self._resume.argtypes = [ctypes.c_uint64]
        self._resume.restype = ctypes.c_int

    @classmethod
    def from_environment(cls) -> "Roctx | None":
        if os.environ.get(ROCTX_ENV_VAR) != "1":
            return None
        path = ctypes.util.find_library("rocprofiler-sdk-roctx")
        if path is None:
            raise RuntimeError(
                f"{ROCTX_ENV_VAR}=1 but librocprofiler-sdk-roctx is unavailable"
            )
        return cls(ctypes.CDLL(path))

    def push(self, name: str) -> None:
        if "\x00" in name:
            raise ValueError("ROCTX range names cannot contain NUL bytes")
        if self._push(name.encode("utf-8")) != 0:
            raise RuntimeError("roctxRangePushA failed")

    def pop(self) -> None:
        if self._pop() != 0:
            raise RuntimeError("roctxRangePop failed")

    def start(self, name: str) -> int:
        """Start a range that may be stopped from another host thread."""
        if "\x00" in name:
            raise ValueError("ROCTX range names cannot contain NUL bytes")
        return int(self._start(name.encode("utf-8")))

    def stop(self, range_id: int) -> None:
        self._stop(range_id)

    def profiler_pause(self) -> None:
        """Stop collection globally when rocprofv3 uses selected regions."""
        if self._pause(0) != 0:
            raise RuntimeError("roctxProfilerPause failed")

    def profiler_resume(self) -> None:
        """Start collection globally when rocprofv3 uses selected regions."""
        if self._resume(0) != 0:
            raise RuntimeError("roctxProfilerResume failed")
