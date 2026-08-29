# SPDX-License-Identifier: Apache-2.0
"""Report which vLLM module pulls in xgrammar, without letting it load.

xgrammar's extension aborts the process on import in this image, so the usual
traceback never appears. Blocking the import turns the abort into an exception
whose stack names the importer.
"""

from __future__ import annotations

import builtins
import sys

_real_import = builtins.__import__


def _traced(name, globals_=None, locals_=None, fromlist=(), level=0):
    if name.split(".")[0] == "xgrammar":
        importers = []
        frame = sys._getframe(1)
        while frame is not None:
            module = frame.f_globals.get("__name__")
            if module and "importlib" not in module and module not in importers:
                importers.append(module)
            frame = frame.f_back
        print("CHAIN:", " <- ".join(importers[:12]), flush=True)
        raise ImportError("xgrammar import blocked for tracing")
    return _real_import(name, globals_, locals_, fromlist, level)


builtins.__import__ = _traced

target = sys.argv[1] if len(sys.argv) > 1 else "vllm.v1.core.kv_cache_utils"
try:
    __import__(target)
    print("RESULT: imported with no xgrammar dependency")
except BaseException as exc:  # noqa: BLE001 - reporting whatever surfaced
    print(f"RESULT: {type(exc).__name__}: {exc}")
