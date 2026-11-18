# SPDX-License-Identifier: Apache-2.0
"""The paged KV runtime, in three layers with a one-way import rule.

`common` is the neutral vocabulary and imports only the standard library and
numpy. `control` is the page allocator and prefix index, and may also import
`common`. `execution` holds the device side: pools, step views, bucketing and
the drivers, and is the only layer that reaches jax or a kernel provider.
`tests/test_kv_import_rule.py` enforces the rule.

Nothing is imported here, so importing `common` or `control` never loads jax.
"""
