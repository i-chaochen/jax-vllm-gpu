# SPDX-License-Identifier: Apache-2.0
"""Kernel providers for the paged KV runtime.

`paged_attention` builds a `PagedPlan` from the step metadata and runs KV
append and paged attention through jax-aiter, with tensor parallelism handled
under `shard_map`.
"""
