# SPDX-License-Identifier: Apache-2.0
"""vLLM platform plugin and KV connector for JAX on AMD GPUs.

Nothing is imported here on purpose. Both entry points name their own module,
so importing this package must stay cheap and must not pull in vllm or jax.
"""

__version__ = "0.0.1.dev0"
