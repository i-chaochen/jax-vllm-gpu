# SPDX-License-Identifier: Apache-2.0
"""Nothing here imports the copy of the paged KV runtime that MaxText still ships.

The pinned MaxText carries the runtime this repository moved out of it, under
the same module names, so an import of `maxtext.inference.kv_execution` resolves
and runs -- just not this repository's code. A plan built by one copy then fails
the other's isinstance check, and a test written against MaxText's copy goes on
passing when this one breaks. Checked over the text of every Python file, so an
import inside a subprocess script string is caught too.
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
SCANNED = ("src", "tests", "tools", "benchmarks")

# MaxText modules whose code now lives in this package.
MOVED = (
    "maxtext.inference.kv_common",
    "maxtext.inference.kv_control",
    "maxtext.inference.kv_execution",
    "maxtext.inference.hf_tokenizer",
    "maxtext.inference.paged_offline_worker",
    "maxtext.layers.gpu_paged_attention",
)

_DOTTED = re.compile(r"\b(?:from|import)\s+(" + "|".join(re.escape(m) for m in MOVED) + r")\b")
_FROM_PARENT = re.compile(
    r"\bfrom\s+(maxtext\.inference|maxtext\.layers)\s+import\s+\(?([\w\s,]+)"
)


def _offences(text: str) -> list[str]:
    found = [m.group(1) for m in _DOTTED.finditer(text)]
    for match in _FROM_PARENT.finditer(text):
        parent = match.group(1)
        for name in re.split(r"[\s,]+", match.group(2)):
            if f"{parent}.{name}" in MOVED:
                found.append(f"{parent}.{name}")
    return found


def _sources() -> list[pathlib.Path]:
    return sorted(p for root in SCANNED for p in (REPO / root).rglob("*.py") if p != pathlib.Path(__file__))


def test_no_file_imports_a_runtime_module_maxtext_still_ships():
    offending = {
        str(path.relative_to(REPO)): hits for path in _sources() if (hits := _offences(path.read_text()))
    }

    assert not offending, (
        f"{offending} import MaxText's copy of a module this repository owns; import it from "
        "vllm_xla_gpu instead (kv.common, kv.control, kv.execution, host, engine, "
        "ops.paged_attention)"
    )


def test_the_scan_recognizes_each_spelling():
    """The check above passes just as well on a scanner that matches nothing."""
    assert _offences("from maxtext.inference.kv_execution.bucketing import X") == [
        "maxtext.inference.kv_execution"
    ]
    assert _offences("import maxtext.layers.gpu_paged_attention as gpa") == [
        "maxtext.layers.gpu_paged_attention"
    ]
    assert _offences('"from maxtext.inference import hf_tokenizer\\n"') == ["maxtext.inference.hf_tokenizer"]
    assert _offences("from maxtext.layers import (attentions, gpu_paged_attention)") == [
        "maxtext.layers.gpu_paged_attention"
    ]
    assert _offences("from maxtext.inference import inference_utils") == []
    assert _offences("from maxtext.inference.maxengine import maxengine") == []
