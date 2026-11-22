# Copyright 2023–2025 Google LLC
# Modifications Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MaxEngine with paged serving: the native engine, with no vLLM in the process.

`PagedEngine` takes model construction, parameter loading, the tokenizer and
the dense path from MaxText's `MaxEngine` unchanged, and adds a parallel paged
surface: `init_paged_runtime`, `prefill_paged`, `generate_paged`, `paged_step`
and `release`. The one `MaxEngine` method it overrides is `release_pages`,
which is a stub there. The dense two-region cache -- one fixed slot per request, every
slot advancing in lockstep, a cache copied into a batch row at insert time --
does not survive contact with a page pool, so the paged path sits beside it
rather than inside it, and nothing on the dense path calls into it.

The paged step threads the pool through MaxText's NNX model and back out.
"""

from __future__ import annotations

import functools
import uuid
from typing import Any

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL
from maxtext.inference import inference_utils
from maxtext.inference.maxengine import maxengine
from maxtext.utils import max_logging

from vllm_xla_gpu.engine.config import PagedConfig
from vllm_xla_gpu.models.adapters import maxtext as maxtext_adapter


class PagedEngine(maxengine.MaxEngine):
  """A MaxEngine whose requests hold pages of a shared pool instead of a slot.

  `paged` sizes the pool. Without it the engine serves only MaxText's dense path.
  """

  def __init__(self, config: Any, devices: Any = None, *, paged: PagedConfig | None = None):
    maxtext_adapter.install()
    self.paged = paged.resolve(config) if paged is not None else None
    super().__init__(config, devices)
    # Set by `init_paged_runtime`. Left None so that `release`/`release_pages`
    # degrade to a log line on every other attention mode rather than requiring
    # a paged runtime to exist.
    self.paged_runtime = None

  def release(self, request_handle: Any, token_ids=None):
    """Reclaim the pages a request holds. The canonical release API.

    Request-based rather than slot-based on purpose. A slot is the dense cache's
    unit -- one fixed reservation held for a request's whole life -- whereas a
    paged request owns a set of pages that changed size on every step, so the
    handle is its only durable name.

    `token_ids` is the request's full context, offered to the prefix cache when
    one is enabled. Pages it adopts stay allocated for the next request with the
    same prefix, and are therefore *not* among the ids returned -- the return
    value continues to mean "free now, safe to overwrite". Omitting the tokens
    publishes nothing, which is the right default for a caller that has not
    thought about cache identity.

    Returns the page ids reclaimed, or an empty array when no paged runtime is
    attached.
    """
    if self.paged_runtime is None:
      max_logging.log("release: paged attention is not configured, so there are no pages to release.")
      return np.empty((0,), dtype=np.int32)
    return self.paged_runtime.release(request_handle, token_ids)

  def release_pages(self, slot: int):
    """`release` for callers that only have a slot.

    Implementing the paged runtime around this signature instead would have
    baked the dense cache's one-slot-per-request assumption into it.
    """
    if self.paged_runtime is None:
      max_logging.log(f"release_pages: slot {slot} ignored because paged attention is not configured.")
      return np.empty((0,), dtype=np.int32)
    return self.paged_runtime.release_slot(slot)

  def init_paged_runtime(
      self,
      max_requests: int | None = None,
      max_batched_tokens: int | None = None,
  ):
    """Allocate the paged pool and control plane. The sibling of `init_decode_state`.

    Args:
      max_requests: concurrent requests to size the page-map rows for. Defaults
        to the dense path's batch width, so a paged deployment is no narrower
        than the dense one it replaces.
      max_batched_tokens: token budget for one prefill step, and the top of the
        token bucket ladder. Defaults to the per-request context limit.

    Returns:
      The `PagedRuntime`, also stored on `self.paged_runtime`.
    """
    # pylint: disable=import-outside-toplevel
    from vllm_xla_gpu.kv.control import NativeKvControlPlane
    from vllm_xla_gpu.kv.execution.bucketing import StepShapePlanner
    from vllm_xla_gpu.kv.execution.engine_adapter import PagedRuntime
    from vllm_xla_gpu.kv.execution.pool_factory import allocate_pool

    if self.paged is None:
      raise ValueError("init_paged_runtime needs the engine built with paged=PagedConfig(...).")

    layout = self._paged_layout()
    if layout.num_kv_heads and layout.kv_head_shards > layout.num_kv_heads:
      # Over-sharding the head axis itself is the one layout with nowhere to go:
      # `kv_pool_sharding` has to split the tensor axis into a head-selecting and
      # a replicating part to place the copies, and it does that on a mesh of its
      # own, while `shard_map` needs the pool and the activations on one mesh.
      #
      # It should already be unreachable, because MaxText refuses to build a
      # model whose KV heads are sharded more ways than it has heads -- attention
      # heads are atomic under tensor parallelism. Checked anyway, because this
      # counts mesh axes directly while MaxText counts them through the logical
      # axis rules, and the two could disagree.
      #
      # Note what is deliberately *not* refused: a replicated KV footprint
      # reached by putting surplus parallelism on an axis that does not shard KV
      # heads, which is the only route MaxText permits. `tensor=4, fsdp=2` on
      # eight devices with four KV heads gives every device one head and pairs of
      # devices the same head, entirely on MaxText's own mesh, and matches the
      # dense path token for token.
      raise ValueError(
          f"attention='gpu_paged' cannot shard KV heads more ways than there are heads: this mesh "
          f"shards them {layout.kv_head_shards} ways over {layout.num_kv_heads} heads. Reduce the "
          f"tensor-parallel width to at most num_kv_heads and put any surplus parallelism on an axis "
          f"that does not shard KV heads, such as fsdp."
      )
    if layout.dtype not in ("bfloat16", "float16"):
      raise ValueError(
          f"the paged attention kernels accept bfloat16 and float16 only, but dtype is "
          f"'{layout.dtype}'. Set dtype=bfloat16 for attention='gpu_paged'."
      )

    max_requests = int(max_requests or self.max_concurrent_decodes)
    control_plane = NativeKvControlPlane(
        layout=layout,
        max_requests=max_requests,
        max_context_len=self.paged.max_context_len,
        enable_prefix_cache=self.paged.enable_prefix_cache,
    )
    planner = StepShapePlanner(
        tokens_per_page=layout.tokens_per_page,
        max_batch=max_requests,
        max_context_len=self.paged.max_context_len,
        pool_pages=layout.num_pages,
        max_batched_tokens=max_batched_tokens,
    )
    self.paged_runtime = PagedRuntime(
        control_plane=control_plane,
        pool=allocate_pool(layout, sharding=self.kv_pool_sharding()),
        planner=planner,
        poison_on_free=self.paged.poison_freed_pages,
    )
    return self.paged_runtime

  def _paged_layout(self):
    # pylint: disable=import-outside-toplevel
    from vllm_xla_gpu.kv.execution.layout_builder import build_storage_layout

    return build_storage_layout(
        self.config,
        self._mesh,
        tokens_per_page=self.paged.page_size,
        usable_pages=self.paged.num_blocks,
    )

  def kv_pool_sharding(self):
    """Sharding for one layer's pool array, or None while unsharded.

    Pages are the leading axis and are never sharded -- a page belongs to one
    request and splitting it would put half a token's KV on another device. The
    KV-head axis is the one tensor parallelism divides, and `kv_pool_sharding`
    works out which head belongs on which device, including the case where TP
    exceeds the KV head count and heads are replicated rather than divided.
    """
    # pylint: disable=import-outside-toplevel
    from vllm_xla_gpu.kv.execution.layout_builder import kv_pool_sharding

    if self.paged is None:
      return None
    return kv_pool_sharding(self._mesh, self._paged_layout())

  def _paged_pool_arrays(self):
    """The pool as the nested list `kv_caches` expects: one `[k, v]` per layer."""
    pool = self.paged_runtime.pool
    return [[pool.k_pages[i], pool.v_pages[i]] for i in range(pool.num_layers)]

  def _rebind_paged_pool(self, updated):
    """Store the aliased pool handles returned by a step.

    Not optional. The pool is donated into the step, so the arrays passed in are
    invalid afterwards and only these handles refer to live memory.
    """
    pool = self.paged_runtime.pool
    for layer, pair in enumerate(updated):
      pool.replace_layer(layer, pair[0], pair[1])

  def _run_paged_model(self, params, *, tokens, positions, segment_ids, model_mode, kv_caches, plan):
    """One NNX forward pass with the pool threaded through. Returns `(logits, kv_caches)`.

    The returned pool is not optional bookkeeping: the pool is donated and comes
    back aliased, so a caller that does not rebind it is holding a deleted array.
    """
    cache_state = self._nnx_cache_state_template(mode=model_mode)
    nnx.replace_by_pure_dict(cache_state, self._nnx_init_cache_dict(mode=model_mode))
    # Merge with the graphdef built for this mode. Layers that captured their
    # model_mode at construction (e.g. the DeepSeek layers) would otherwise run
    # the prefill attention path during autoregressive decode.
    graphdef = self.graphdef_ar if model_mode == MODEL_MODE_AUTOREGRESSIVE else self.graphdef
    # copy=True avoids reusing Variable objects across traces (TraceContextError).
    model = nnx.merge(graphdef, params, cache_state, self._nnx_rest_state, copy=True)
    return model(
        tokens,
        positions,
        decoder_segment_ids=segment_ids,
        enable_dropout=False,
        model_mode=model_mode,
        kv_caches=kv_caches,
        attention_metadata=plan,
    )

  @functools.partial(
      jax.jit,
      static_argnums=(0,),
      static_argnames=("model_mode", "max_seqlen_k", "is_decode", "algorithm", "topk", "nucleus_topp"),
      donate_argnames=("kv_caches",),
  )
  def _paged_forward_jit(
      self,
      *,
      params: Any,
      kv_caches,
      tokens: jax.Array,
      positions: jax.Array,
      segment_ids: jax.Array | None,
      slot_mapping: jax.Array,
      kv_indptr: jax.Array,
      kv_page_indices: jax.Array,
      kv_last_page_lens: jax.Array,
      cu_seqlens_q: jax.Array,
      sample_rows: jax.Array,
      sample_at: jax.Array,
      rng: Any,
      model_mode: str,
      max_seqlen_k: int,
      is_decode: bool,
      algorithm: str | None = None,
      topk: int | None = None,
      nucleus_topp: float | None = None,
      temperature: float | None = None,
  ):
    """One paged step: write this step's K/V into the pool, attend, and sample.

    The page bookkeeping arrives as plain int32 arrays rather than as a
    `PagedPlan`, because the plan is not a pytree and could not cross a `jit`
    boundary; it is reassembled here where the arrays are already tracers.

    `sample_at` is the per-row index whose logits to sample, traced rather than
    static so a new prompt length does not force a new trace.
    """
    # pylint: disable=import-outside-toplevel
    from vllm_xla_gpu.ops.paged_attention import PagedPlan

    plan = PagedPlan(
        slot_mapping=slot_mapping,
        kv_indptr=kv_indptr,
        kv_page_indices=kv_page_indices,
        kv_last_page_lens=kv_last_page_lens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1 if is_decode else int(tokens.shape[1]),
        max_seqlen_k=max_seqlen_k,
        is_decode=is_decode,
    )

    # The first split is the forward pass's key. The NNX forward takes none, but
    # splitting anyway keeps the sampled tokens for a given seed unchanged.
    rng, _ = jax.random.split(rng)
    with self._mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      logits, updated_pools = self._run_paged_model(
          params,
          tokens=tokens,
          positions=positions,
          segment_ids=segment_ids,
          model_mode=model_mode,
          kv_caches=kv_caches,
          plan=plan,
      )

    # One logit row per request, gathered at its own (row, position). A gather
    # rather than a slice is what lets a decode batch hold requests at different
    # positions, which is the whole point of paging.
    #
    # Both indices are supplied because the two phases lay a batch out along
    # different axes: decode batches along rows and samples position 0 of each,
    # while prefill *packs* requests into one row and samples several positions
    # within it. Deriving the rows as `arange(logits.shape[0])` is correct for
    # decode and silently returns a single token for a packed multi-request
    # prefill.
    selected = logits[sample_rows, sample_at][:, None, :]
    selected = jax.lax.with_sharding_constraint(selected, self.replicated_sharding)

    rng, sample_rng = jax.random.split(rng)
    sampled = inference_utils.sampling(
        selected,
        sample_rng,
        algorithm if algorithm is not None else self.config.decode_sampling_strategy,
        topk=topk if topk is not None else self.config.decode_sampling_top_k,
        nucleus_topp=nucleus_topp if nucleus_topp is not None else self.config.decode_sampling_nucleus_p,
        temperature=temperature if temperature is not None else self.config.decode_sampling_temperature,
    )
    return sampled, selected, updated_pools

  def paged_step(
      self,
      *,
      params: Any,
      view,
      inputs,
      is_decode: bool,
      rng: Any = None,
      algorithm: str | None = None,
      topk: int | None = None,
      nucleus_topp: float | None = None,
      temperature: float | None = None,
  ):
    """One paged forward pass over an already-reserved step. No policy, no admission.

    The seam the scheduler plugs into. `view` is the page bookkeeping from
    `PagedRuntime.prepare_step`; `inputs` is the token side from
    `build_step_inputs`. Everything this needs has already been decided, which is
    what makes it usable by both `PagedDriver` -- which has admitted and reserved
    itself -- and by `prefill_paged` / `generate_paged`, which wrap it in their own
    admission and bookkeeping.

    Returns `(sampled, selected)`: the tokens, and the logit rows they came from.
    The logits are returned rather than dropped because `return_log_prob` needs
    them, and building `ResultTokens` stays with the callers since it is a
    JetStream-shaped concern the driver has no use for.

    The pool is rebound as a side effect, since the forward pass donates it.
    """
    if rng is None:
      if self.rng is None:
        self.rng = jax.random.PRNGKey(0)
      self.rng, rng = jax.random.split(self.rng)

    sampled, selected, updated = self._paged_forward_jit(
        params=params,
        kv_caches=self._paged_pool_arrays(),
        tokens=jnp.asarray(inputs.tokens),
        positions=jnp.asarray(inputs.positions),
        segment_ids=None if inputs.segment_ids is None else jnp.asarray(inputs.segment_ids),
        slot_mapping=view.slot_mapping,
        kv_indptr=view.kv_indptr,
        kv_page_indices=view.kv_page_indices,
        kv_last_page_lens=view.kv_last_page_lens,
        cu_seqlens_q=view.cu_seqlens_q,
        sample_rows=jnp.asarray(inputs.sample_rows),
        sample_at=jnp.asarray(inputs.sample_at),
        rng=rng,
        model_mode=MODEL_MODE_AUTOREGRESSIVE if is_decode else MODEL_MODE_PREFILL,
        max_seqlen_k=view.shape.max_seqlen_k,
        is_decode=is_decode,
        algorithm=algorithm,
        topk=topk,
        nucleus_topp=nucleus_topp,
        temperature=temperature,
    )
    self._rebind_paged_pool(updated)
    return sampled, selected

  def paged_step_fn(self, params: Any, **kwargs):
    """A `PagedDriver` step function bound to this engine.

    The driver's contract is `(view, inputs, pool) -> tokens`. The pool arrives
    for implementations that need it; this one does not, because the engine
    already holds the pool it donates and rebinds.
    """

    def step(view, inputs, pool):
      del pool
      sampled, _ = self.paged_step(params=params, view=view, inputs=inputs, is_decode=view.shape.is_decode, **kwargs)
      return np.asarray(sampled).reshape(-1)[: inputs.sample_at.size]

    return step

  def _paged_result_tokens(self, sampled: jax.Array, selected: jax.Array, step: int):
    """Wrap sampled tokens in the same `ResultTokens` shape the dense path returns."""
    all_valid = jnp.ones(sampled.shape, dtype=jnp.int8)
    lengths = jnp.full(sampled.shape, step, dtype=jnp.int32)
    if self.config.return_log_prob:
      token_logp = inference_utils.log_prob_of_chosen_token(selected, sampled)
    else:
      token_logp = jnp.zeros(sampled.shape, dtype=jnp.float32)
    return maxengine.engine_api.ResultTokens(
        data=jnp.concatenate((sampled, all_valid, lengths), axis=1),
        tokens_idx=(0, 1),
        valid_idx=(1, 2),
        length_idx=(2, 3),
        log_prob=token_logp,
        samples_per_slot=1,
    )

  def prefill_paged(
      self,
      *,
      params: Any,
      padded_tokens: jax.Array,
      true_length: int,
      request_id: str | None = None,
      max_new_tokens: int | None = None,
      slot: int | None = None,
      prompt_token_ids=None,
      namespace=None,
      rng: Any = None,
      algorithm: str | None = None,
      topk: int | None = None,
      nucleus_topp: float | None = None,
      temperature: float | None = None,
  ):
    """Prefill one prompt into the page pool and sample its first token.

    The sibling of `prefill`. It returns a `RequestHandle` rather than a prefix
    dict, because there is no cache to carry: the K/V is already in the pool and
    the handle is what names the pages holding it.

    Passing `prompt_token_ids` opts the request into prefix sharing, when the
    control plane has it enabled. The prompt's already-computed leading pages are
    then lent to the request and only the remainder is run.

    Returns:
      `(handle, result_tokens)`, or `(None, None)` if the pool could not admit
      the request -- backpressure, not an error.
    """
    if self.paged_runtime is None:
      raise ValueError("call init_paged_runtime() before prefill_paged()")

    runtime = self.paged_runtime
    prompt_len = int(true_length)
    budget = self.paged.max_context_len - prompt_len
    handle = runtime.admit(
        request_id=request_id if request_id is not None else f"req-{uuid.uuid4()}",
        prompt_len=prompt_len,
        max_new_tokens=int(max_new_tokens) if max_new_tokens is not None else max(budget, 0),
    )
    if handle is None:
      return None, None

    cached = runtime.attach_prefix(handle, prompt_token_ids, namespace)
    query_len = prompt_len - cached
    view = runtime.prepare_step([handle], [query_len], is_decode=False, num_requests=1)
    if view is None:
      runtime.control_plane.release(handle)
      return None, None

    # pylint: disable=import-outside-toplevel
    from vllm_xla_gpu.kv.execution.step_inputs import RequestSlice, build_step_inputs

    # Built in numpy, deliberately. The `jnp` spelling compiles a fresh program
    # per prompt length: the input to a pad is a different shape each time, and
    # `arange(n) < prompt_len` bakes `prompt_len` into the jaxpr as a literal.
    #
    # With a prefix hit the query is the prompt's suffix, so the tokens are
    # sliced from `cached` and the positions start there too. `start=cached` is
    # what makes the suffix rotate as a suffix; RoPE is not translation
    # invariant, and getting it wrong yields plausible text rather than an error.
    prompt = np.asarray(padded_tokens, dtype=np.int64).reshape(-1)
    inputs = build_step_inputs(
        [RequestSlice(tokens=prompt[cached : cached + query_len], start=cached, query_len=query_len)],
        view.shape,
        is_decode=False,
    )
    sampled, selected = self.paged_step(
        params=params,
        view=view,
        inputs=inputs,
        is_decode=False,
        rng=rng,
        algorithm=algorithm,
        topk=topk,
        nucleus_topp=nucleus_topp,
        temperature=temperature,
    )
    runtime.track(handle, slot=slot)
    return handle, self._paged_result_tokens(sampled, selected, step=0)

  def generate_paged(
      self,
      params: Any,
      handles,
      next_tokens: jax.Array,
      rng: Any = None,
      algorithm: str | None = None,
      topk: int | None = None,
      nucleus_topp: float | None = None,
      temperature: float | None = None,
  ):
    """Advance every live request by one token. The sibling of `generate`.

    Unlike the dense path there is no lockstep: each request contributes one
    token at its own position, which is read from the control plane rather than
    from a shared counter.

    Returns:
      `(result_tokens, ok)`. `ok` is False when the pool could not back the step,
      which the caller resolves by releasing or preempting something.
    """
    if self.paged_runtime is None:
      raise ValueError("call init_paged_runtime() before generate_paged()")

    runtime = self.paged_runtime
    handles = list(handles)
    if not handles:
      raise ValueError("generate_paged needs at least one live request")

    # Read before reserving: this is each request's position for the token about
    # to be written, and reserving advances the recorded length past it.
    positions = [int(runtime.control_plane.page_map.seq_len(h)) for h in handles]

    view = runtime.prepare_step(handles, [1] * len(handles), is_decode=True)
    if view is None:
      return None, False

    # pylint: disable=import-outside-toplevel
    from vllm_xla_gpu.kv.execution.step_inputs import RequestSlice, build_step_inputs

    supplied = np.asarray(next_tokens, dtype=np.int64).reshape(-1)[: len(handles)]
    inputs = build_step_inputs(
        [
            RequestSlice(tokens=np.asarray([token], np.int64), start=position, query_len=1)
            for position, token in zip(positions, supplied.tolist())
        ],
        view.shape,
        is_decode=True,
    )
    sampled, selected = self.paged_step(
        params=params,
        view=view,
        inputs=inputs,
        is_decode=True,
        rng=rng,
        algorithm=algorithm,
        topk=topk,
        nucleus_topp=nucleus_topp,
        temperature=temperature,
    )
    return self._paged_result_tokens(sampled, selected, step=1), True
