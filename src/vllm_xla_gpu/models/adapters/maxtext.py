# SPDX-License-Identifier: Apache-2.0
"""MaxText models, with paged attention served by this repository's kernels.

MaxText's `Attention` sends `attention="gpu_paged"` to its
`forward_serve_gpu_paged` method. `install()` binds that method to the
implementation below, which builds the plan and runs the kernels from
`vllm_xla_gpu.ops`. Call it before building a model.
"""

from __future__ import annotations

from typing import Any

HOOK = "forward_serve_gpu_paged"


def forward_serve_gpu_paged(
    self,
    query: Any,
    key: Any,
    value: Any,
    kv_pools: list[Any] | None = None,
    metadata: Any = None,
) -> tuple[Any, list[Any]]:
  """Paged attention for one serving step, bound onto MaxText's `Attention`.

  `kv_pools` is `[k_pool, v_pool]`, each NHD
  `[num_pages, tokens_per_page, num_kv_heads, head_dim]`. The step writes this
  token's K/V into the pool and then attends over it, so prefill and decode
  read exactly the pages append wrote.
  """
  # pylint: disable=import-outside-toplevel
  from vllm_xla_gpu.ops import paged_attention

  query = query.reshape(-1, query.shape[2], query.shape[3]).astype(self.dtype)
  key = key.reshape(-1, key.shape[2], key.shape[3]).astype(self.dtype)
  value = value.reshape(-1, value.shape[2], value.shape[3]).astype(self.dtype)

  if not kv_pools or metadata is None:
    # Dry run: model initialization and JIT tracing reach here with nothing to
    # attend over. Mirrors forward_serve_vllm; without it, init breaks rather
    # than inference, which is a confusing way to fail.
    return query, []

  k_pool, v_pool = kv_pools[0], kv_pools[1]
  plan = paged_attention.build_plan(
      metadata,
      tokens_per_page=k_pool.shape[1],
      total_tokens=query.shape[0],
      max_seqlen_k=self.max_target_length,
  )
  # `scale=1.0`, not 1/sqrt(head_dim). MaxText folds the depth scaling into the
  # query projection's initializer and applies `query_pre_attn_scalar` before
  # this point, so the query arriving here is already scaled; letting the kernel
  # apply its own default would scale twice.
  #
  # Built once and shared by both branches deliberately. Spelled out separately
  # they drifted: the sharded branch omitted the scale, so every tensor-parallel
  # run scaled the query a second time, flattening the softmax towards uniform.
  # That is a wrong answer rather than a crash, and no comparison of the sharded
  # step against the single-device step can see it, because such a test passes
  # the same scale to both sides.
  kernel_kwargs = dict(backend="auto", scale=1.0, causal=True)
  axes = paged_attention.kv_head_axes(self.mesh)
  if axes:
    # Tensor parallelism: the kernels have to run in manual mode, because XLA
    # cannot partition the FFI call they go through.
    return paged_attention.paged_attention_step_sharded(
        query,
        key,
        value,
        k_pool,
        v_pool,
        plan,
        mesh=self.mesh,
        axes=axes if len(axes) > 1 else axes[0],
        **kernel_kwargs,
    )
  return paged_attention.paged_attention_step(query, key, value, k_pool, v_pool, plan, **kernel_kwargs)


def install() -> None:
  """Bind MaxText's paged-attention hook to `forward_serve_gpu_paged`. Idempotent.

  Raises if MaxText has no such hook, which means the checkout is missing the
  gpu_paged dispatch: an unbound `attention="gpu_paged"` would otherwise fail
  far from the cause.
  """
  from maxtext.layers import attentions  # pylint: disable=import-outside-toplevel

  if not hasattr(attentions.Attention, HOOK):
    raise RuntimeError(
        f"MaxText at {attentions.__file__} has no Attention.{HOOK}, so it cannot dispatch "
        "attention='gpu_paged'. Use a MaxText checkout that carries the gpu_paged hook."
    )
  setattr(attentions.Attention, HOOK, forward_serve_gpu_paged)


def installed() -> bool:
  """Whether MaxText's hook is bound to this module's implementation."""
  from maxtext.layers import attentions  # pylint: disable=import-outside-toplevel

  return getattr(attentions.Attention, HOOK, None) is forward_serve_gpu_paged
