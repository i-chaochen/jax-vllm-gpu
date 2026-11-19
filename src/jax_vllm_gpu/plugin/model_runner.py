# SPDX-License-Identifier: Apache-2.0
"""The model runner: one `SchedulerOutput` in, one `ModelRunnerOutput` out.

Everything host-side that this needs already exists -- `BatchState` accumulates
vLLM's block-table deltas, `prepare_step` runs the scrub-then-build order that
carries the recycled-page guarantee, and `StepView` pads the result onto a
bucketed shape. What is added here is the part that needs a model: building the
pool, calling MaxText, sampling, and reporting.

Three things about the model call are not obvious and are load-bearing.

**MaxText returns logits, not hidden states, in `gpu_paged` mode.** Its
`Transformer.__call__` returns `hidden_state` for the `vllm_rpa` attentions --
because the TPU plugin computes logits separately -- but `logits` for
`gpu_paged`, since MaxText's own serving loop samples inside the same jitted
step. The vLLM adapter names the value `hidden` either way. So the second
element of the adapter's return is this platform's logits, and calling the
adapter's `compute_logits` on it would apply the output head a second time.

**Parameters are split out of the module and passed as an argument.** A closure
capturing the model would bake every weight into the jaxpr as a constant, which
at 70B is not a slow path but an impossible one. `nnx.split` is how
`tpu-inference` avoids the same thing.

**The pool is donated and returned aliased.** The arrays passed in are dead
afterwards; `pool.rebind` is what keeps the next step pointing at live memory.
"""

from __future__ import annotations

import logging
import time
from typing import Any, NamedTuple

import numpy as np

from . import metrics
from . import pool as pool_module
from . import paged_plan, sampling
from .batch_state import BatchState
from .kv_reuse import PageReuseTracker
from .page_table import RequestPages
from .profiling import Roctx
from .step import prepare_step

logger = logging.getLogger(__name__)


class PendingStep(NamedTuple):
    """What `execute_model` leaves for `sample_tokens` to finish.

    `logits` stays a device array. It is already one row per request, so this
    is small, and moving it to the host here would put a synchronisation point
    between the two calls that the batch-queue path exists to avoid.

    `started` is when the step began, so the timing recorded at the end of
    `sample_tokens` covers the whole step. Measuring the two halves separately
    would attribute nothing to `execute_model`, which enqueues asynchronously and
    returns before the work is done. `roctx_range` spans the same interval, for
    the same reason.
    """

    requests: tuple
    logits: Any
    shape: Any
    started: float
    roctx_range: int | None = None


class RequestTokens:
    """A request's token sequence and how it should be sampled.

    Separate from `BatchState`, which tracks pages. vLLM sends prompt tokens
    once, with the new-request record, and never resends them, so the runner has
    to keep them to feed later prefill chunks -- and has to append each sampled
    token, since that token is the next step's input.
    """

    __slots__ = ("request_id", "tokens", "num_prompt_tokens", "sampling_params")

    def __init__(self, request_id: str, prompt_token_ids, sampling_params) -> None:
        self.request_id = request_id
        self.tokens: list[int] = list(prompt_token_ids or ())
        self.num_prompt_tokens = len(self.tokens)
        self.sampling_params = sampling_params

    def slice(self, start: int, count: int) -> np.ndarray:
        end = start + count
        if end > len(self.tokens):
            raise ValueError(
                f"request {self.request_id!r} needs tokens [{start}:{end}) but "
                f"only {len(self.tokens)} are known; a sampled token was "
                "probably not appended"
            )
        return np.asarray(self.tokens[start:end], dtype=np.int32)


class JaxModelRunner:
    """Drives one step of a MaxText model over an externally owned KV pool."""

    def __init__(self, vllm_config: Any, mesh: Any) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.mesh = mesh

        self.batch_state = BatchState()
        self.requests: dict[str, RequestTokens] = {}

        self.model: Any = None
        self.pool: Any = None
        self.layout: Any = None
        self.tracker: PageReuseTracker | None = None
        self.planner: Any = None
        self.scrub: Any = None

        self._graphdef: Any = None
        self._state_treedef: Any = None
        self._state_leaves: Any = None
        self._forward_fn: Any = None
        self._sample_fn: Any = None
        self._rng: Any = None

        # Carries one step's logits from execute_model to sample_tokens.
        self._pending: PendingStep | None = None

        # One ROCTX range per step when JAX_VLLM_ROCTX=1; None loads nothing.
        self._roctx = Roctx.from_environment()
        self._roctx_steps = 0

        self.tokens_per_page = int(self.cache_config.block_size)

    # ---------------------------------------------------------------- model

    def load_model(self) -> None:
        """Build the MaxText model and split its parameters out."""
        import jax
        from flax import nnx

        from .registry import get_model_cls, register_maxtext

        # Idempotent, and the hook can fire more than once per process.
        register_maxtext()
        model_cls = get_model_cls("MaxTextForCausalLM")

        self._rng = jax.random.key(int(self.model_config.seed or 0))
        self._rng, init_rng = jax.random.split(self._rng)

        with jax.set_mesh(self.mesh):
            model = model_cls(self.vllm_config, init_rng, self.mesh)
            # `load_weights` is a no-op once the model exists, which is the case
            # after a dummy-weight init; calling it unconditionally is what
            # makes the checkpoint path and the dummy path the same code.
            model.load_weights(init_rng)

        self.model = model
        graphdef, state = nnx.split(model)
        self._graphdef = graphdef
        self._state_leaves, self._state_treedef = jax.tree_util.tree_flatten(state)
        logger.info(
            "Loaded MaxText model: %d parameter arrays.", len(self._state_leaves)
        )

    def get_model(self) -> Any:
        return self.model

    @property
    def maxtext_config(self) -> Any:
        if self.model is None:
            raise RuntimeError("the model has not been loaded yet")
        return self.model.maxtext_config

    def model_geometry(self) -> tuple[int, int, int]:
        """`(num_layers, num_kv_heads, head_dim)`, as MaxText resolved them.

        Read from MaxText's config rather than the HuggingFace one because the
        adapter pads `base_num_kv_heads` up to the tensor-parallel degree for
        grouped-query models, and a pool sized from the unpadded count would be
        too small by exactly that factor.
        """
        config = self.maxtext_config
        num_layers = int(
            getattr(config, "num_decoder_layers", 0)
            or getattr(config, "base_num_decoder_layers", 0)
        )
        num_kv_heads = int(getattr(config, "num_kv_heads", 0) or 0)
        head_dim = int(getattr(config, "head_dim", 0) or 0)
        if not (num_layers and num_kv_heads and head_dim):
            raise ValueError(
                "MaxText config did not yield a usable KV geometry: "
                f"layers={num_layers}, kv_heads={num_kv_heads}, head_dim={head_dim}"
            )
        return num_layers, num_kv_heads, head_dim

    # ------------------------------------------------------------- kv cache

    def get_kv_cache_spec(self) -> dict[str, Any]:
        from .kv_spec import build_kv_cache_spec

        num_layers, num_kv_heads, head_dim = self.model_geometry()
        return build_kv_cache_spec(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            dtype=self.model_config.dtype,
            block_size=self.tokens_per_page,
        )

    def initialize_kv_cache(self, kv_cache_config: Any) -> None:
        """Allocate the pool for the block count vLLM settled on."""
        from jax_vllm_gpu.kv.execution.bucketing import StepShapePlanner

        from .kv_spec import build_layout
        from .mesh import assert_kv_heads_shardable, kv_head_shards, pool_replicas

        num_layers, num_kv_heads, head_dim = self.model_geometry()
        assert_kv_heads_shardable(self.mesh, num_kv_heads)

        num_blocks = int(kv_cache_config.num_blocks)
        self.layout = build_layout(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype_name=str(getattr(self.maxtext_config, "dtype", "bfloat16")),
            tokens_per_page=self.tokens_per_page,
            num_pages=num_blocks,
            kv_head_shards=kv_head_shards(self.mesh),
            pool_replicas=pool_replicas(self.mesh),
        )

        self.pool = pool_module.allocate(self.layout, self.mesh)
        self.scrub = pool_module.make_scrub_fn(self.pool)
        self.tracker = PageReuseTracker(
            num_pages=num_blocks, tokens_per_page=self.tokens_per_page
        )
        self.planner = StepShapePlanner(
            tokens_per_page=self.tokens_per_page,
            max_batch=int(self.scheduler_config.max_num_seqs),
            max_context_len=int(self.model_config.max_model_len),
            pool_pages=num_blocks,
            max_batched_tokens=int(self.scheduler_config.max_num_batched_tokens),
        )
        self._forward_fn = self._build_forward_fn()
        self._sample_fn = self._build_sample_fn()

    # ------------------------------------------------------------ the step

    def _build_forward_fn(self):
        """Compile the forward pass, up to one row of logits per request.

        `is_decode` and the padded extents are static, so a shape from the
        bucketing ladder selects an existing executable rather than triggering a
        recompile. That is the whole reason the ladder exists: without it every
        distinct token count in a serving trace is a fresh compile of an
        80-layer model.

        Row selection happens inside this trace rather than after it, so what
        crosses into `sample_tokens` is `[num_requests, vocab]` rather than
        `[num_tokens, vocab]`. On a long prefill chunk that is the difference
        between a few megabytes and a few gigabytes.
        """
        import functools

        import jax
        import jax.numpy as jnp
        from flax import nnx

        graphdef = self._graphdef
        treedef = self._state_treedef

        @functools.partial(
            jax.jit,
            static_argnames=("max_seqlen_q", "max_seqlen_k", "is_decode"),
            donate_argnames=("kv_caches",),
        )
        def forward(
            state_leaves,
            kv_caches,
            input_ids,
            input_positions,
            slot_mapping,
            kv_indptr,
            kv_page_indices,
            kv_last_page_lens,
            cu_seqlens_q,
            sample_rows,
            *,
            max_seqlen_q: int,
            max_seqlen_k: int,
            is_decode: bool,
        ):
            model = nnx.merge(graphdef, jax.tree_util.tree_unflatten(treedef, state_leaves))
            metadata = paged_plan.PagedPlanWithPositions(
                slot_mapping=slot_mapping,
                kv_indptr=kv_indptr,
                kv_page_indices=kv_page_indices,
                kv_last_page_lens=kv_last_page_lens,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                is_decode=is_decode,
                input_positions=input_positions,
            )

            # Second element is logits on this path, despite the adapter naming
            # it `hidden`; see the module docstring.
            kv_caches, logits, _aux, _experts = model(kv_caches, input_ids, metadata)
            return kv_caches, jnp.take(logits, sample_rows, axis=0)

        return forward

    def _build_sample_fn(self):
        """Compile sampling, separately from the forward pass.

        Separate because vLLM samples in a separate call: `step_with_batch_queue`
        hands the engine `sample_tokens`'s future and discards `execute_model`'s,
        so the tokens have to be produced there. Keeping the logits on device
        between the two makes that split free rather than a round trip.
        """
        import functools

        import jax

        @functools.partial(jax.jit, static_argnames=("all_greedy",))
        def sample(rng, logits, temperature, top_p, top_k, *, all_greedy: bool):
            return sampling.sample(rng, logits, (temperature, top_p, top_k), all_greedy)

        return sample

    def execute_model(self, scheduler_output: Any) -> Any:
        """Run the forward pass and stash its logits. Returns None by design.

        Returning None is what tells vLLM to call `sample_tokens` next, and it
        is not optional. `step_with_batch_queue` queues the *sample_tokens*
        future and reads the `ModelRunnerOutput` from that, discarding whatever
        this returned -- so a runner that sampled here and reported completion
        would have its output silently dropped on that path. The other path,
        plain `step`, calls `sample_tokens` only when this returns None. None
        therefore satisfies both, and anything else satisfies at most one.
        """
        import jax
        import jax.numpy as jnp

        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

        started = time.perf_counter()
        if self._pending is not None:
            raise RuntimeError(
                "execute_model was called twice without sample_tokens in "
                "between; the first step's logits would be discarded"
            )

        # Retire first: a finished request's id may be reused by a new one in
        # the same scheduler output.
        self._retire(scheduler_output)
        self._admit(scheduler_output)
        self.batch_state.apply(scheduler_output)

        if not getattr(scheduler_output, "total_num_scheduled_tokens", 0):
            # Nothing to stash, so nothing for sample_tokens to do. Returned
            # rather than None because on the batch-queue path an unexecuted
            # step's future is this one.
            return EMPTY_MODEL_RUNNER_OUTPUT

        requests = self.batch_state.pages_for_step(scheduler_output)
        shape = self._shape_for(requests)

        prepared = prepare_step(
            requests,
            tracker=self.tracker,
            scrub=self.scrub,
            shape=shape,
            tokens_per_page=self.tokens_per_page,
            token_ids=self._tokens_for,
            padding_page_id=self.layout.padding_page_id,
        )

        view = prepared.view
        input_ids, positions, sample_rows = _prepared_model_inputs(
            prepared.inputs,
            is_decode=shape.is_decode,
            active_requests=len(requests),
            request_width=shape.num_requests,
        )

        roctx_range = None
        if self._roctx is not None:
            self._roctx_steps += 1
            roctx_range = self._roctx.start(
                f"JAX_VLLM_STEP index={self._roctx_steps} decode={int(shape.is_decode)} "
                f"requests={len(requests)} "
                f"active_tokens={sum(r.num_scheduled_tokens for r in requests)} "
                f"tokens={shape.num_tokens} max_seqlen_k={shape.max_seqlen_k}"
            )

        with jax.set_mesh(self.mesh):
            updated, logits = self._forward_fn(
                self._state_leaves,
                pool_module.as_kv_caches(self.pool),
                jnp.asarray(input_ids),
                jnp.asarray(positions),
                view.slot_mapping,
                view.kv_indptr,
                view.kv_page_indices,
                view.kv_last_page_lens,
                view.cu_seqlens_q,
                jnp.asarray(sample_rows),
                max_seqlen_q=_attention_max_seqlen_q(shape),
                max_seqlen_k=shape.max_seqlen_k,
                is_decode=shape.is_decode,
            )
        pool_module.rebind(self.pool, updated)

        self._pending = PendingStep(
            requests=requests,
            logits=logits,
            shape=shape,
            started=started,
            roctx_range=roctx_range,
        )
        return None

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        """Sample from the stashed logits and report the step.

        Reached with nothing stashed whenever `execute_model` had no tokens to
        run, which the engine does not treat as an error, so it answers with an
        empty output rather than raising.
        """
        import jax
        import jax.numpy as jnp

        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

        pending, self._pending = self._pending, None
        if pending is None:
            return EMPTY_MODEL_RUNNER_OUTPUT

        if grammar_output is not None:
            raise NotImplementedError(
                "structured output is not supported yet; the grammar bitmask "
                "would have to mask the logits before this sample"
            )

        requests, shape = pending.requests, pending.shape
        spec = sampling.from_params(
            [self.requests[r.request_id].sampling_params for r in requests],
            vocab_size=int(self.model_config.get_vocab_size()),
        )

        self._rng, step_rng = _next_rng(
            self._rng, all_greedy=spec.all_greedy, split=jax.random.split
        )
        with jax.set_mesh(self.mesh):
            tokens = self._sample_fn(
                step_rng,
                pending.logits,
                jnp.asarray(_pad(spec.temperature, shape.num_requests, 0.0)),
                jnp.asarray(_pad(spec.top_p, shape.num_requests, 1.0)),
                jnp.asarray(_pad(spec.top_k, shape.num_requests, 1)),
                all_greedy=spec.all_greedy,
            )

        # The device_get is the step's only synchronisation point, so timing has
        # to stop after it: everything before returns while the work is still in
        # flight.
        sampled = np.asarray(jax.device_get(tokens), dtype=np.int64)
        if pending.roctx_range is not None:
            self._roctx.stop(pending.roctx_range)
        emitted = self._collect(requests, sampled)
        metrics.STEPS.record(
            shape=shape,
            seconds=time.perf_counter() - pending.started,
            num_requests=len(requests),
            tokens=sum(len(row) for row in emitted),
        )

        return ModelRunnerOutput(
            req_ids=[r.request_id for r in requests],
            req_id_to_index={r.request_id: i for i, r in enumerate(requests)},
            sampled_token_ids=emitted,
        )

    # --------------------------------------------------------------- pieces

    def _admit(self, scheduler_output: Any) -> None:
        if self.tracker is None:
            raise RuntimeError("the KV cache must be initialized before admitting requests")
        for new in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            self.tracker.admit(new.req_id)
            self.requests[new.req_id] = RequestTokens(
                new.req_id, new.prompt_token_ids, new.sampling_params
            )

    def _retire(self, scheduler_output: Any) -> None:
        for finished in getattr(scheduler_output, "finished_req_ids", ()) or ():
            self.requests.pop(finished, None)
            if self.tracker is not None:
                self.tracker.retire(finished)

    def _tokens_for(self, request: RequestPages) -> np.ndarray:
        return self.requests[request.request_id].slice(
            request.num_computed_tokens, request.num_scheduled_tokens
        )

    def _shape_for(self, requests) -> Any:
        query_lens = [r.num_scheduled_tokens for r in requests]
        max_seq_len = max((r.seq_len for r in requests), default=1)
        if all(length == 1 for length in query_lens):
            return self.planner.decode_shape(len(requests), max_seq_len)
        return self.planner.extend_shape(
            sum(query_lens), max_seq_len, num_requests=len(requests)
        )

    def _collect(self, requests, sampled: np.ndarray) -> list[list[int]]:
        """One sampled token per request, or none while it is still prefilling.

        Under chunked prefill a request is scheduled for several steps before
        its prompt is consumed, and the token predicted from a partial prompt is
        not a continuation of anything. vLLM reads an empty list as "no output
        this step", which is exactly the case.
        """
        out: list[list[int]] = []
        for row, request in enumerate(requests):
            state = self.requests[request.request_id]
            if request.seq_len < state.num_prompt_tokens:
                out.append([])
                continue
            token = int(sampled[row])
            state.tokens.append(token)
            out.append([token])
        return out


def _pad(values: np.ndarray, width: int, fill) -> np.ndarray:
    """Right-pad to `width`, which the bucketing ladder guarantees is enough."""
    values = np.asarray(values)
    if values.shape[0] > width:
        raise ValueError(
            f"cannot pad {values.shape[0]} entries down to {width}; the "
            "bucketing ladder is mis-sized"
        )
    if values.shape[0] == width:
        return values
    tail = np.full((width - values.shape[0],), fill, dtype=values.dtype)
    return np.concatenate([values, tail])


def _prepared_model_inputs(
    inputs: Any, *, is_decode: bool, active_requests: int, request_width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The step's flat token ids, positions and sample rows, from `prepare_step`'s host arrays.

    Flat `[num_tokens]`: MaxText's vLLM adapter treats vLLM's flattened tokens
    as a batch of length-one sequences. A sample row is the flat index of a
    request's last token, which is its row under decode (one token per row) and
    its packed position under prefill. Read from host arrays because
    `StepView.cu_seqlens_q` is a device array, and deriving the rows from it
    copied it back to the host on every step.
    """
    input_ids = np.asarray(inputs.tokens, dtype=np.int32).reshape(-1)
    positions = np.asarray(inputs.positions, dtype=np.int32).reshape(-1)
    if is_decode:
        rows = np.asarray(inputs.sample_rows[:active_requests], dtype=np.int32)
    else:
        rows = np.asarray(inputs.sample_at, dtype=np.int32)
    return input_ids, positions, _pad(rows, request_width, 0)


def _attention_max_seqlen_q(shape: Any) -> int:
    """A static bound on any one request's query length in this step.

    The token bucket is the sum over requests, not the longest query; passed as
    is, an 8,192-token mixed batch looked to the paged-prefill kernel like one
    8,192-token query. No query is longer than the bucket or than its own
    context, which `max_seqlen_k` bounds.
    """
    if shape.is_decode:
        return 1
    return min(int(shape.num_tokens), int(shape.max_seqlen_k))


def _next_rng(rng: Any, *, all_greedy: bool, split: Any) -> tuple[Any, Any]:
    """`(next_key, step_key)`, without dispatching a split greedy sampling cannot use."""
    if all_greedy:
        return rng, rng
    return split(rng)
