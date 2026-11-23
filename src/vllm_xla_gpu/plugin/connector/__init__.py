# SPDX-License-Identifier: Apache-2.0
"""KV connector registration, reached through the `vllm.general_plugins` group.

The connector has no entry point group of its own: registration is a plain
classmethod on `KVConnectorFactory`, so something has to call it. The
`vllm.general_plugins` group is the right host because it loads in every
process, including the workers that actually move pages.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

CONNECTOR_NAME = "JaxConnector"
_CONNECTOR_MODULE = "vllm_xla_gpu.plugin.connector.connector"
_CONNECTOR_CLASS = "JaxConnector"


def register() -> None:
    """Register the KV connector. Safe to call more than once.

    `general_plugins` is documented to load in every process and may load more
    than once within one, while `register_connector` raises on a duplicate
    name, so swallowing that specific failure is required rather than sloppy.
    """
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    try:
        KVConnectorFactory.register_connector(
            CONNECTOR_NAME, _CONNECTOR_MODULE, _CONNECTOR_CLASS
        )
    except ValueError:
        logger.debug("%s was already registered; nothing to do.", CONNECTOR_NAME)
        return

    logger.info("Registered KV connector %s.", CONNECTOR_NAME)
