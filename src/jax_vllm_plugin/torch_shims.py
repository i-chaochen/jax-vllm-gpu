# SPDX-License-Identifier: Apache-2.0
"""Make `torch.accelerator` survive a torch that has no accelerator.

vLLM guards its accelerator calls with `not current_platform.is_cpu()`, and
`is_cpu()` tests `_enum`. This platform is `PlatformEnum.OOT` -- deliberately,
because being read as the CPU platform would put vLLM on its CPU execution
paths -- so those guards let the calls through, and a CPU-only torch then raises
from the C extension.

That is not a mistake in either place so much as a case neither anticipated: an
out-of-tree platform whose accelerator is real but is not torch's. `tpu-inference`
patches the same function for the same reason, which is the precedent followed
here.

**Kept narrow on purpose.** Each wrapper swallows exactly the message that means
"torch has no accelerator" and re-raises everything else, so a genuine allocator
failure still surfaces instead of being hidden behind a shim installed for an
unrelated reason.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# What torch's C extension raises when no accelerator is registered at all.
_NO_ACCELERATOR = "Cannot access accelerator device when none is available"

_installed = False


def _wrap(module, name: str) -> None:
    original = getattr(module, name, None)
    if original is None:
        return

    def patched(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except RuntimeError as error:
            if _NO_ACCELERATOR not in str(error):
                raise
            logger.debug("torch.accelerator.%s is a no-op here: %s", name, error)
            return None

    setattr(module, name, patched)


def install() -> None:
    """Tolerate the accelerator calls vLLM makes during teardown.

    `cleanup_dist_env_and_memory` calls `empty_cache` on any platform that is
    not vLLM's CPU one. It runs from `EngineCore.shutdown`, so without this
    every run ends in a traceback after its output has already been produced --
    harmless in itself, and exactly the kind of noise that hides a real
    shutdown failure later.

    Idempotent, because `general_plugins` may load more than once per process.
    """
    global _installed
    if _installed:
        return

    import torch

    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None:
        return

    for name in ("empty_cache", "empty_host_cache", "synchronize"):
        _wrap(accelerator, name)
    _installed = True
