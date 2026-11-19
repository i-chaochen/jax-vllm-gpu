# SPDX-License-Identifier: Apache-2.0
"""The connector's registration hook has to tolerate being called twice."""

from __future__ import annotations

import pytest

from jax_vllm_gpu.plugin.connector import CONNECTOR_NAME, register

pytest.importorskip("vllm", reason="vLLM is supplied by the runtime image")


def test_registration_is_idempotent():
    """vllm.general_plugins loads in every process and may load more than once
    in one, while KVConnectorFactory.register_connector raises on a duplicate
    name. Without this, the second load crashes the worker."""
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    register()
    register()

    assert CONNECTOR_NAME in KVConnectorFactory._registry
