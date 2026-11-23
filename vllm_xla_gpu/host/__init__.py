# SPDX-License-Identifier: Apache-2.0
"""What a step needs before it reaches a device: its shape and its inputs.

`shapes` is the ladder of step shapes, so a churning batch traces a fixed set
of them. `step_inputs` is the tokens, the positions and where to sample. Both
are plain numpy on the host, and both the vLLM plugin and the standalone engine
use them.

This package imports only the standard library, numpy and `kv.common`, the rule
`kv.common` and `kv.control` follow, and `tests/test_kv_import_rule.py` enforces
it. A worker that runs its steps with no JAX in the process can import it.

Nothing is imported here, so importing one module never loads the other.
"""
