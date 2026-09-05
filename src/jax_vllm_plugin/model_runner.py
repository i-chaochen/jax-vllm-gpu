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

import json
import logging
import os
import re
import sys
import time
from typing import Any

import numpy as np

from . import pool as pool_module
from . import paged_plan, sampling
from .batch_state import BatchState
from .kv_reuse import PageReuseTracker
from .page_table import RequestPages
from .profiling import Roctx
from .step import prepare_step

logger = logging.getLogger(__name__)

_RUN_REQUEST_RE = re.compile(
    r"^(?:(?:chat)?cmpl-)?(?P<phase>warmup|measure)(?P<rep>[1-9][0-9]*)-"
)


def _run_identity(requests) -> tuple[str, int | None]:
    """Return a conservative benchmark phase/repetition label."""
    labels: set[tuple[str, int]] = set()
    unknown = False
    for request in requests:
        match = _RUN_REQUEST_RE.match(request.request_id)
        if match is None:
            unknown = True
            continue
        phase = "measured" if match.group("phase") == "measure" else "warmup"
        labels.add((phase, int(match.group("rep"))))
    if not labels:
        return "unknown", None
    if unknown or len(labels) != 1:
        return "cross_run", None
    return next(iter(labels))


def _scheduled_token_split(requests, token_states) -> dict[str, int | str]:
    """Split scheduled work at each request's prompt boundary."""
    prompt_tokens = 0
    output_tokens = 0
    prefill_requests = 0
    decode_requests = 0
    for request in requests:
        prompt_end = token_states[request.request_id].num_prompt_tokens
        start = request.num_computed_tokens
        end = start + request.num_scheduled_tokens
        prompt = max(0, min(end, prompt_end) - min(start, prompt_end))
        output = request.num_scheduled_tokens - prompt
        prompt_tokens += prompt
        output_tokens += output
        prefill_requests += int(prompt > 0)
        decode_requests += int(output > 0)

    if prompt_tokens and output_tokens:
        batch_kind = "mixed"
    elif prompt_tokens:
        batch_kind = "prefill"
    else:
        batch_kind = "decode"
    return {
        "batch_kind": batch_kind,
        "prefill_requests": prefill_requests,
        "decode_requests": decode_requests,
        "scheduled_prompt_tokens": prompt_tokens,
        "scheduled_output_tokens": output_tokens,
        "scheduled_tokens": prompt_tokens + output_tokens,
    }


def _prepared_model_inputs(
    inputs,
    *,
    is_decode: bool,
    active_requests: int,
    request_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reuse MaxText's host inputs without a device round trip.

    `prepare_step` already builds the padded token, position, and sampling
    arrays. Reconstructing them from `StepView` both duplicated that work and,
    because `StepView.cu_seqlens_q` is a JAX array, copied metadata back from
    the device only to send the derived sample rows to the device again.
    """
    input_ids = np.asarray(inputs.tokens, dtype=np.int32).reshape(-1)
    input_positions = np.asarray(inputs.positions, dtype=np.int32).reshape(-1)
    if is_decode:
        active_sample_rows = np.asarray(
            inputs.sample_rows[:active_requests], dtype=np.int32
        )
    else:
        active_sample_rows = np.asarray(inputs.sample_at, dtype=np.int32)
    sample_rows = _pad(active_sample_rows, request_width, 0)
    return input_ids, input_positions, sample_rows


def _attention_max_seqlen_q(shape) -> int:
    """Return a tight static upper bound for any query segment.

    The token bucket is the sum across requests, not the longest request.
    Passing that sum as `max_seqlen_q` made an 8,192-token mixed batch look
    like one 8,192-token query to the paged-prefill kernel. A request's query
    cannot exceed either the total token bucket or its post-step context bound.
    """
    if shape.is_decode:
        return 1
    return min(int(shape.num_tokens), int(shape.max_seqlen_k))


def _densify_token_rungs(rungs) -> tuple[int, ...]:
    """Insert midpoints below the largest power-of-two interval.

    JAX needs static token extents, but powers of two are not the only bounded
    policy. Midpoints cap padding at roughly 1.5x while preserving a small,
    deterministic executable set. The largest interval stays coarse: a
    near-maximum bucket is expensive to compile, is sensitive to request-arrival
    timing, and saved no padded work in the matched production trace.
    """
    base = tuple(int(rung) for rung in rungs)
    if len(base) < 2:
        return base
    dense: list[int] = []
    for lower, upper in zip(base[:-2], base[1:-1], strict=True):
        dense.append(lower)
        midpoint = lower + (upper - lower) // 2
        if lower < midpoint < upper:
            dense.append(midpoint)
    dense.extend(base[-2:])
    return tuple(dense)


def _next_rng(rng, *, all_greedy: bool, split):
    """Do not dispatch a random split for a program that cannot use randomness."""
    if all_greedy:
        return rng, rng
    return split(rng)


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
        self._step_fn: Any = None
        self._rng: Any = None
        self._log_shapes = os.environ.get("JAX_VLLM_LOG_SHAPES") == "1"
        self._observed_shapes: set[Any] = set()
        self._step_telemetry = os.environ.get("JAX_VLLM_STEP_TELEMETRY") == "1"
        self._step_telemetry_index = 0
        self._roctx = Roctx.from_environment()
        self._roctx_step_index = 0
        self._xplane_dir = os.environ.get("JAX_VLLM_XPLANE_DIR")
        self._xplane_expected_tokens = int(
            os.environ.get("JAX_VLLM_XPLANE_EXPECTED_TOKENS", "0")
        )
        if self._xplane_dir and self._xplane_expected_tokens <= 0:
            raise ValueError(
                "JAX_VLLM_XPLANE_EXPECTED_TOKENS must be positive when "
                "JAX_VLLM_XPLANE_DIR is set"
            )
        self._xplane_started = False
        self._xplane_done = False
        self._xplane_tokens = 0

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

        with self.mesh:
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
        from maxtext.inference.kv_execution.bucketing import StepShapePlanner

        from .kv_spec import build_layout
        from .mesh import assert_kv_heads_shardable, kv_head_shards, pool_replicas

        num_layers, num_kv_heads, head_dim = self.model_geometry()
        assert_kv_heads_shardable(self.maxtext_config, self.mesh, num_kv_heads)

        num_blocks = int(kv_cache_config.num_blocks)
        self.layout = build_layout(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype_name=str(getattr(self.maxtext_config, "dtype", "bfloat16")),
            tokens_per_page=self.tokens_per_page,
            num_pages=num_blocks,
            kv_head_shards=kv_head_shards(self.maxtext_config, self.mesh),
            pool_replicas=pool_replicas(self.maxtext_config, self.mesh),
        )

        self.pool = pool_module.allocate(self.layout, self.mesh, self.maxtext_config)
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
        self.planner.token_rungs = _densify_token_rungs(self.planner.token_rungs)
        logger.info("Serving token buckets: %s.", self.planner.token_rungs)
        self._step_fn = self._build_step_fn()

    # ------------------------------------------------------------ the step

    def _build_step_fn(self):
        """Compile the forward-plus-sample step.

        `is_decode` and the padded extents are static, so a shape from the
        bucketing ladder selects an existing executable rather than triggering a
        recompile. That is the whole reason the ladder exists: without it every
        distinct token count in a serving trace is a fresh compile of an
        80-layer model.
        """
        import functools

        import jax
        import jax.numpy as jnp
        from flax import nnx

        graphdef = self._graphdef
        treedef = self._state_treedef

        @functools.partial(
            jax.jit,
            static_argnames=("max_seqlen_q", "max_seqlen_k", "is_decode", "all_greedy"),
            donate_argnames=("kv_caches",),
        )
        def step(
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
            temperature,
            top_p,
            top_k,
            rng,
            *,
            max_seqlen_q: int,
            max_seqlen_k: int,
            is_decode: bool,
            all_greedy: bool,
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
            kv_caches, logits, _aux, _experts = model(
                kv_caches, input_ids, metadata
            )
            selected = jnp.take(logits, sample_rows, axis=0)
            tokens = sampling.sample(
                rng, selected, (temperature, top_p, top_k), all_greedy
            )
            return kv_caches, tokens

        return step

    def execute_model(self, scheduler_output: Any) -> Any:
        import jax
        import jax.numpy as jnp

        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

        step_start_ns = time.monotonic_ns() if self._step_telemetry else 0
        self._retire(scheduler_output)
        self._admit(scheduler_output)
        self.batch_state.apply(scheduler_output)

        if not getattr(scheduler_output, "total_num_scheduled_tokens", 0):
            return EMPTY_MODEL_RUNNER_OUTPUT

        requests = self.batch_state.pages_for_step(scheduler_output)
        shape = self._shape_for(requests)
        max_seqlen_q = _attention_max_seqlen_q(shape)
        if self._log_shapes and shape not in self._observed_shapes:
            self._observed_shapes.add(shape)
            request_id = requests[0].request_id if requests else "-"
            logger.warning(
                "JAX_VLLM_NEW_SHAPE index=%d request=%s active_requests=%d "
                "active_tokens=%d shape=%s",
                len(self._observed_shapes),
                request_id,
                len(requests),
                sum(r.num_scheduled_tokens for r in requests),
                shape,
            )

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
        spec = sampling.from_params(
            [self.requests[r.request_id].sampling_params for r in requests],
            vocab_size=int(self.model_config.get_vocab_size()),
        )

        self._rng, step_rng = _next_rng(
            self._rng,
            all_greedy=spec.all_greedy,
            split=jax.random.split,
        )
        profile_metadata = None
        if self._roctx is not None or (self._xplane_dir and not self._xplane_done):
            self._roctx_step_index += 1
            run_phase, run_rep = _run_identity(requests)
            token_split = _scheduled_token_split(requests, self.requests)
            profile_metadata = {
                "phase": run_phase,
                "rep": run_rep or 0,
                "step": self._roctx_step_index,
                "kind": token_split["batch_kind"],
                "active_requests": len(requests),
                "true_tokens": token_split["scheduled_tokens"],
                "padded_tokens": shape.num_tokens,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": shape.max_seqlen_k,
            }
        if self._roctx is not None:
            self._roctx.push(
                "JAX_VLLM_MODEL_CALL "
                + " ".join(f"{key}={value}" for key, value in profile_metadata.items())
            )
        xplane_call = bool(
            self._xplane_dir
            and not self._xplane_done
            and profile_metadata["phase"] == "measured"
            and profile_metadata["rep"] == 1
        )
        if xplane_call and not self._xplane_started:
            jax.profiler.start_trace(
                self._xplane_dir,
                create_perfetto_trace=True,
                create_perfetto_link=False,
            )
            self._xplane_started = True
        annotation = None
        if xplane_call:
            annotation = jax.profiler.TraceAnnotation(
                "JAX_VLLM_MODEL_CALL",
                **profile_metadata,
            )
            annotation.__enter__()
        execute_start_ns = time.monotonic_ns() if self._step_telemetry else 0
        try:
            with self.mesh:
                updated, tokens = self._step_fn(
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
                    jnp.asarray(_pad(spec.temperature, shape.num_requests, 0.0)),
                    jnp.asarray(_pad(spec.top_p, shape.num_requests, 1.0)),
                    jnp.asarray(_pad(spec.top_k, shape.num_requests, 1)),
                    step_rng,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=shape.max_seqlen_k,
                    is_decode=shape.is_decode,
                    all_greedy=spec.all_greedy,
                )
            pool_module.rebind(self.pool, updated)
            sampled = np.asarray(jax.device_get(tokens), dtype=np.int64)
        finally:
            if annotation is not None:
                annotation.__exit__(*sys.exc_info())
            if self._roctx is not None:
                self._roctx.pop()
        if xplane_call:
            self._xplane_tokens += int(profile_metadata["true_tokens"])
            if self._xplane_tokens > self._xplane_expected_tokens:
                raise RuntimeError(
                    "XPlane capture exceeded its expected true-token total: "
                    f"{self._xplane_tokens} > {self._xplane_expected_tokens}"
                )
            if self._xplane_tokens == self._xplane_expected_tokens:
                jax.profiler.stop_trace()
                self._xplane_done = True
        execute_end_ns = time.monotonic_ns() if self._step_telemetry else 0
        if self._step_telemetry:
            self._step_telemetry_index += 1
            run_phase, run_rep = _run_identity(requests)
            token_split = _scheduled_token_split(requests, self.requests)
            query_lens = np.asarray(prepared.table.query_lens)
            seq_lens = np.asarray(prepared.table.seq_lens)
            record = {
                "schema": 1,
                "event": "jax_vllm_step",
                "step_index": self._step_telemetry_index,
                "run_phase": run_phase,
                "run_rep": run_rep,
                "active_requests": len(requests),
                "resident_requests": len(self.batch_state),
                **token_split,
                "true_shape": {
                    "num_requests": int(prepared.view.num_active_requests),
                    "num_tokens": int(prepared.view.num_active_tokens),
                    "num_pages": sum(len(row) for row in prepared.table.page_ids),
                    "max_seqlen_q": int(query_lens.max()) if query_lens.size else 0,
                    "max_seqlen_k": int(seq_lens.max()) if seq_lens.size else 0,
                },
                "padded_shape": {
                    "num_requests": int(shape.num_requests),
                    "num_tokens": int(shape.num_tokens),
                    "num_pages": int(shape.num_pages),
                    "max_seqlen_q": max_seqlen_q,
                    "max_seqlen_k": int(shape.max_seqlen_k),
                    "is_decode": bool(shape.is_decode),
                },
                "timing_ns": {
                    "start_monotonic": step_start_ns,
                    "prepare": execute_start_ns - step_start_ns,
                    "execute_blocking": execute_end_ns - execute_start_ns,
                    "timed_total": execute_end_ns - step_start_ns,
                },
            }
            logger.warning(
                "JAX_VLLM_STEP %s",
                json.dumps(record, sort_keys=True, separators=(",", ":")),
            )
        return ModelRunnerOutput(
            req_ids=[r.request_id for r in requests],
            req_id_to_index={r.request_id: i for i, r in enumerate(requests)},
            sampled_token_ids=self._collect(requests, sampled),
        )

    # --------------------------------------------------------------- pieces

    def _admit(self, scheduler_output: Any) -> None:
        if self.tracker is None:
            raise RuntimeError("KV cache must be initialized before admitting requests")
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
        return self.planner.extend_shape(sum(query_lens), max_seq_len)

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
