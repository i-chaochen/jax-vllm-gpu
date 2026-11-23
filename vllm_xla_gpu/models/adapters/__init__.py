# SPDX-License-Identifier: Apache-2.0
"""Adapters from model providers to the paged KV runtime.

`maxtext` binds MaxText's paged-attention hook to this repository's kernels and
is shared by the vLLM plugin and the standalone engine. `maxtext_vllm` wraps a
MaxText model as the `MaxTextForCausalLM` that the vLLM plugin serves.
"""
