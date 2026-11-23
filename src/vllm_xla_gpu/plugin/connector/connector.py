# SPDX-License-Identifier: Apache-2.0
"""KV connector for disaggregation and host offload. Not implemented yet.

Registration is lazy -- `KVConnectorFactory` stores a loader and imports this
module only when a connector is actually constructed -- so registering the name
succeeds today and instantiation is what fails.

When implemented this will subclass
`vllm.distributed.kv_transfer.kv_connector.base.KVConnectorBase`. Two references
apply, and neither is a dependency: `tpu_inference.distributed.tpu_connector` is
the structural precedent for a JAX-backed connector, since vLLM's own connector
API assumes torch tensors; and vLLM's in-tree `moriio` connector is the protocol
precedent for driving MORI, though it is torch-bound and so cannot be reused
against XLA-owned pool arrays.
"""

from __future__ import annotations


class JaxConnector:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "JaxConnector is not implemented yet. It depends on the M9 page "
            "transfer op, and on choosing between the host-side mori.io API and "
            "the XLA FFI custom call over MORI's device-side one-sided puts."
        )
