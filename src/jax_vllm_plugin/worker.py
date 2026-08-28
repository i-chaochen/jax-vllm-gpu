# SPDX-License-Identifier: Apache-2.0
"""Worker entry point, named by `JaxPlatform.check_and_update_config`.

Not implemented yet. It is referenced by qualname rather than imported, so
platform discovery succeeds without it and fails here instead -- which is the
intended split: the plugin is discoverable now, and executing a request is
honest about being unfinished.
"""

from __future__ import annotations


class JaxWorker:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "JaxWorker is not implemented yet. It will own the JAX model "
            "runner, physical KV pool allocation via pool_factory, bucketing "
            "and precompile."
        )
