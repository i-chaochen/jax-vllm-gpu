# SPDX-License-Identifier: Apache-2.0
"""Under EXTERNAL_VLLM there must be exactly one page owner, and it is vLLM.

Section 4 allows two ownership modes and insists only one allocator is live at a
time. That is easy to satisfy by accident today and easy to break by accident
later: importing `NativeKvControlPlane` to reuse one helper would put a second
allocator in the process, and nothing about the output would look wrong -- both
allocators would hand out plausible page ids, from disjoint free lists, for the
same pool.

So this asserts inertness structurally rather than behaviourally. A test that
merely ran a step and saw the right answer would pass just as happily with a
second allocator sitting unused beside the first.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import vllm_xla_gpu.plugin

# The page-ownership half of the standalone control plane, and the standalone
# engine that drives it. `kv_execution` is deliberately not here: its pool
# factory, step view and pool ops are frontend-agnostic and this platform is a
# legitimate user of all of them. It is specifically the allocator and the
# prefix index that vLLM replaces, and the engine's driver and runtime reach
# both.
FORBIDDEN_ROOTS = ("vllm_xla_gpu.kv.control", "vllm_xla_gpu.engine")


def plugin_modules():
    """Every source file in the package, `__init__` and subpackages included.

    Globbed rather than walked with `pkgutil`, which skips a package's own
    `__init__.py` -- and a rule about what may not be imported is worth nothing
    if it does not cover every file that can import.
    """
    root = Path(vllm_xla_gpu.plugin.__file__).parent
    return sorted(root.rglob("*.py"))


def imported_names(path: Path):
    """Every module named by an import in `path`, including inside functions.

    Walking the AST rather than importing, because a lazy import inside a
    function body is exactly the way this rule would get broken without anyone
    noticing at module load.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module


@pytest.mark.parametrize("path", plugin_modules(), ids=lambda p: p.name)
def test_no_plugin_module_reaches_maxtexts_own_allocator(path):
    offending = sorted(
        {
            name
            for name in imported_names(path)
            for root in FORBIDDEN_ROOTS
            if name == root or name.startswith(f"{root}.")
        }
    )

    assert not offending, (
        f"{path.name} imports {offending}. Under EXTERNAL_VLLM vLLM's "
        "KVCacheManager is the sole page owner; the standalone allocator and prefix "
        "index are bypassed, not merely unused."
    )


def test_the_scan_actually_sees_imports():
    """The rule above would pass just as well on a scanner that found nothing.

    Anchored on `kv_execution`, which this platform legitimately uses and whose
    name differs from the forbidden one by a single path component -- so this
    also pins that the match is on the component and not on a loose prefix.
    """
    seen = {name for path in plugin_modules() for name in imported_names(path)}

    assert any(name.startswith("vllm_xla_gpu.kv.execution") for name in seen)


def test_the_scrub_contract_still_has_an_answerer():
    """The positive half, and the reason the tracker is shaped as it is.

    `KvControlPlane` declares `pending_scrub` and `confirm_scrubbed` as part of
    the protocol rather than as implementation detail, precisely so that a
    frontend owning its own page accounting is still forced to answer them.
    Bypassing the allocator therefore has to leave those two questions answered
    by something, and here it is `PageReuseTracker`.
    """
    from vllm_xla_gpu.kv.control.protocols import KvControlPlane

    from vllm_xla_gpu.plugin.kv_reuse import PageReuseTracker

    scrub_contract = ("pending_scrub", "confirm_scrubbed")
    for name in scrub_contract:
        assert hasattr(KvControlPlane, name), f"{name} left the control-plane protocol"
        assert callable(getattr(PageReuseTracker, name, None)), (
            f"nothing in EXTERNAL_VLLM mode answers {name}"
        )
