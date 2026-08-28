# SPDX-License-Identifier: Apache-2.0
"""Model registry for this platform.

`tpu_inference` plays this role for the TPU plugin, and MaxText's adapter used to
import its `register_model` directly, which is what tied the adapter to TPU. The
adapter now takes the registry as an argument, so this module supplies ours.

The dependency runs plugin -> maxtext, never the reverse: MaxText must not import
this package.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Any] = {}


def register_model(name: str, model_cls: Any) -> None:
    """Register a model class under `name`. Re-registering the same class is
    tolerated because plugin hooks can load more than once per process."""
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not model_cls:
        raise ValueError(
            f"{name!r} is already registered to {existing!r}, refusing to "
            f"replace it with {model_cls!r}"
        )
    _REGISTRY[name] = model_cls
    logger.info("Registered model %s.", name)


def get_model_cls(name: str) -> Any:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no model registered as {name!r}; registered names are "
            f"{sorted(_REGISTRY)}"
        ) from None


def registered_models() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def register_maxtext(registry: Callable[[str, Any], None] | None = None) -> None:
    """Register MaxText's adapter against this platform's registry.

    Imported lazily: MaxText pulls in JAX and a good deal of config, and a
    deployment that never serves a MaxText model should not pay for it.
    """
    from maxtext.integration.vllm.maxtext_vllm_adapter import register as register_adapter

    register_adapter(register_model=registry or register_model)
