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
from threading import Lock
from typing import Any

import numpy as np

from . import pool as pool_module
from . import paged_plan, sampling
from .batch_state import BatchState
from .kv_reuse import PageReuseTracker
from .page_table import RequestPages
from .profiling import (
    ROCTX_EXPECTED_TOKENS_ENV_VAR,
    ROCTX_PROFILE_REP_ENV_VAR,
    ROCTX_SELECTED_REGIONS_ENV_VAR,
    Roctx,
)
from .step import prepare_step

logger = logging.getLogger(__name__)
TOKEN_BUCKET_SUBDIVISIONS_ENV_VAR = "JAX_VLLM_TOKEN_BUCKET_SUBDIVISIONS"
ATTENTION_SPLIT_ENV_VAR = "JAX_VLLM_ATTENTION_SPLIT"
# Allocates the x-vectorized KV pool, which routes decode to AITER's ASM kernel.
# Off by default: it is the pool *layout* that selects the kernel, so this
# changes how KV is stored, and it gives up the `direct_i32` append fast path.
ASM_DECODE_ENV_VAR = "JAX_VLLM_ASM_DECODE"
# Must agree with MaxText's models.defer_output_head(): one variable read
# on both sides, never inferred separately in two places that could disagree.
DEFER_LOGITS_ENV_VAR = "MAXTEXT_DEFER_LOGITS"

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


def _pack_step_operands(
    input_ids,
    input_positions,
    view,
    sample_rows,
    temperature,
    top_p,
    top_k,
    *,
    chained_token_sources=None,
    request_token_slots=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Coalesce small host arrays into one transfer per dtype."""
    if (chained_token_sources is None) != (request_token_slots is None):
        raise ValueError(
            "chained token sources and request token slots must be provided together"
        )
    int_operands = [
        input_ids,
        input_positions,
        view.slot_mapping,
        view.kv_indptr,
        view.kv_page_indices,
        view.kv_last_page_lens,
        view.cu_seqlens_q,
        sample_rows,
        top_k,
    ]
    block_tables = getattr(view, "block_tables", None)
    context_lens = getattr(view, "seq_lens", None)
    if block_tables is not None:
        if context_lens is None:
            raise ValueError(
                "a dense block table requires full context lengths"
            )
        # Keep the page map in the one int32 host-to-device transfer rather
        # than adding two small transfers per model call. The jitted step knows
        # the static table width from the runner's pool layout.
        int_operands.extend((block_tables, context_lens))
    if chained_token_sources is not None:
        int_operands.extend((chained_token_sources, request_token_slots))
    packed_ints = np.concatenate(
        [
            np.asarray(values, dtype=np.int32).reshape(-1)
            for values in int_operands
        ]
    )
    packed_floats = np.concatenate(
        [
            np.asarray(values, dtype=np.float32).reshape(-1)
            for values in (temperature, top_p)
        ]
    )
    return packed_ints, packed_floats


def _unpack_step_operands(
    packed_ints,
    packed_floats,
    *,
    num_tokens: int,
    num_requests: int,
    num_pages: int,
    block_table_blocks: int | None = None,
    include_chaining: bool = False,
):
    """Slice packed operands using the static extents of one executable."""
    int_offset = 0

    def take_int(size: int):
        nonlocal int_offset
        result = packed_ints[int_offset : int_offset + size]
        int_offset += size
        return result

    input_ids = take_int(num_tokens)
    input_positions = take_int(num_tokens)
    slot_mapping = take_int(num_tokens)
    kv_indptr = take_int(num_requests + 1)
    kv_page_indices = take_int(num_pages)
    kv_last_page_lens = take_int(num_requests)
    cu_seqlens_q = take_int(num_requests + 1)
    sample_rows = take_int(num_requests)
    top_k = take_int(num_requests)
    if block_table_blocks is not None:
        block_tables = take_int(num_requests * block_table_blocks).reshape(
            num_requests, block_table_blocks
        )
        context_lens = take_int(num_requests)
    else:
        block_tables = None
        context_lens = None
    chained_token_sources = take_int(num_tokens) if include_chaining else None
    request_token_slots = take_int(num_requests) if include_chaining else None
    if int_offset != packed_ints.shape[0]:
        raise ValueError(
            f"packed int operand has {packed_ints.shape[0]} values, expected {int_offset}"
        )

    temperature = packed_floats[:num_requests]
    top_p = packed_floats[num_requests : 2 * num_requests]
    if packed_floats.shape[0] != 2 * num_requests:
        raise ValueError(
            f"packed float operand has {packed_floats.shape[0]} values, "
            f"expected {2 * num_requests}"
        )
    operands = (
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
    )
    if block_table_blocks is not None:
        operands += (block_tables, context_lens)
    if include_chaining:
        operands += (chained_token_sources, request_token_slots)
    return operands


# How many leading request rows a split mixed step reserves for chunk-prefilling
# requests. Measured on the matched replay: 1 request prefills in 64.7% of mixed
# steps, 2 in 19.5%, 3 in 9.8%, 4 in 2.3%, so eight slots cover 97.1% and the
# remaining 2.9% take the unsplit path unchanged. Larger is not free -- every
# reserved slot is a row the prefill kernel processes whether or not it holds a
# request.
PREFILL_SLOTS = 8


def count_prefill(requests) -> int:
    """How many leading requests are chunk-prefilling.

    Counts the *prefix*, not the total, because the split slices the batch at a
    fixed row: a single-token request sitting before a multi-token one means the
    batch was never sorted, and silently counting past it would hand the prefill
    kernel a row whose tokens live in the decode half.
    """
    count = 0
    for request in requests:
        if request.num_scheduled_tokens == 1:
            break
        count += 1
    return count


def _attention_max_seqlen_q(shape) -> int:
    """Return a tight static upper bound for any query segment.

    The token bucket is the sum across requests, not the longest request.
    Passing that sum as `max_seqlen_q` made an 8,192-token mixed batch look
    like one 8,192-token query to the paged-prefill kernel. A request's query
    cannot exceed either the total token bucket or its post-step context bound.
    """
    if shape.is_decode:
        return 1
    # Under the split, the prefill kernel receives only `query[:num_prefill_tokens]`
    # -- the decode rows go to the decode kernel -- so bounding it by the combined
    # width would declare a query extent about one decode batch longer than the
    # slice that is actually passed.
    # NOTE: passing the *prefill extent* here instead of the combined width is
    # analytically the tighter bound -- the split hands the prefill kernel only
    # `query[:num_prefill_tokens]` -- but it is measured 3x SLOWER end to end
    # (5,758 -> 1,796 tok/s; mixed steps 8.4 -> 118 ms/call). Shrinking the bound
    # moves the ck_tile batch-prefill kernel onto a different, far worse
    # instantiation. Opt in only to re-measure that.
    extent = shape.num_tokens
    if (
        os.environ.get("JAX_VLLM_PREFILL_EXTENT_Q", "0") == "1"
        and getattr(shape, "num_prefill_slots", None) is not None
    ):
        extent = shape.num_prefill_tokens
    return min(int(extent), int(shape.max_seqlen_k))


def _densify_token_rungs(rungs, subdivisions: int = 2) -> tuple[int, ...]:
    """Subdivide every interval except the largest one.

    JAX needs static token extents, but powers of two are not the only bounded
    policy. Two subdivisions preserve the prior midpoint policy; four cap
    per-interval padding at roughly 1.25x. The largest interval stays coarse: a
    near-maximum bucket is expensive to compile, is sensitive to request-arrival
    timing, and saved no padded work in the matched production trace.
    """
    if subdivisions < 1:
        raise ValueError(f"token bucket subdivisions must be positive, got {subdivisions}")
    base = tuple(int(rung) for rung in rungs)
    if len(base) < 2:
        return base
    dense: list[int] = []
    for lower, upper in zip(base[:-2], base[1:-1], strict=True):
        for part in range(subdivisions):
            rung = lower + (upper - lower) * part // subdivisions
            if not dense or rung > dense[-1]:
                dense.append(rung)
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

    __slots__ = (
        "request_id",
        "tokens",
        "num_prompt_tokens",
        "sampling_params",
        "device_token_slot",
    )

    def __init__(
        self,
        request_id: str,
        prompt_token_ids,
        sampling_params,
        device_token_slot: int = -1,
    ) -> None:
        self.request_id = request_id
        self.tokens: list[int] = list(prompt_token_ids or ())
        self.num_prompt_tokens = len(self.tokens)
        self.sampling_params = sampling_params
        self.device_token_slot = device_token_slot

    def slice(self, start: int, count: int) -> np.ndarray:
        end = start + count
        if end > len(self.tokens):
            raise ValueError(
                f"request {self.request_id!r} needs tokens [{start}:{end}) but "
                f"only {len(self.tokens)} are known; a sampled token was "
                "probably not appended"
            )
        return np.asarray(self.tokens[start:end], dtype=np.int32)


def _pending_token_offset(request: RequestPages, state: RequestTokens) -> int | None:
    """Locate the one token an async step produced but host output has not read."""
    start = int(request.num_computed_tokens)
    count = int(request.num_scheduled_tokens)
    end = start + count
    known = len(state.tokens)
    if end <= known:
        return None
    if start <= known and end == known + 1:
        return known - start
    raise ValueError(
        f"request {request.request_id!r} needs tokens [{start}:{end}) but only "
        f"{known} are known; async chaining can supply exactly one trailing token"
    )


def _chained_token_sources(
    requests,
    token_states,
    *,
    num_tokens: int,
    sample_rows: np.ndarray,
) -> np.ndarray:
    """Map missing input rows to stable slots in the device token carry.

    The target comes from `sample_rows` rather than from a cursor walked over
    `num_scheduled_tokens`. The pending token is always a request's *last*
    scheduled token -- `_pending_token_offset` returns only when
    `end == known + 1` -- and `sample_rows` already holds the flat index of that
    token in whatever layout the packer produced: the row itself when decode
    batches along rows, `sample_at` when prefill packs along one sequence.

    Reconstructing a contiguous cursor instead was wrong under the split
    packing, where decode tokens begin at the padded prefill capacity rather
    than after the true prefill tokens. The chained token landed in the inert
    gap while the real decode position kept its zero placeholder, so a mixed
    async step decoded token id 0.
    """
    sources = np.full((num_tokens,), -1, dtype=np.int32)
    for row, request in enumerate(requests):
        state = token_states[request.request_id]
        offset = _pending_token_offset(request, state)
        if offset is not None:
            if state.device_token_slot < 0:
                raise ValueError(
                    f"request {request.request_id!r} has no device token slot"
                )
            sources[int(sample_rows[row])] = state.device_token_slot
    return sources


class JaxModelRunner:
    """Drives one step of a MaxText model over an externally owned KV pool."""

    # Class-level so partially constructed runners (tests build them with
    # `__new__`) still resolve the split default. `__init__` overrides it from
    # JAX_VLLM_ATTENTION_SPLIT.
    _prefill_slots: int | None = PREFILL_SLOTS
    _defer_logits: bool = False
    # None until `initialize_kv_cache` decides the pool layout, and the value
    # that means "build no dense page map", so the NHD path is unaffected.
    _block_table_blocks: int | None = None

    def __init__(self, vllm_config: Any, mesh: Any) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.mesh = mesh

        self.batch_state = BatchState()
        self.requests: dict[str, RequestTokens] = {}
        self._async_scheduling = bool(self.scheduler_config.async_scheduling)
        self._max_request_slots = int(self.scheduler_config.max_num_seqs)
        self._free_token_slots = list(range(self._max_request_slots - 1, -1, -1))
        self._chained_tokens: Any = None
        self._pending_async_result: Any = None

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
        # `None`, not 0, is the off value: it clears `num_prefill_slots`, which is
        # what `bucketing.extend_shape`, `step_view`, `step_inputs`,
        # `PagedPlan.is_split` and the jit static arg all branch on. A 0 would
        # disable the split but still perturb the request-row width.
        self._prefill_slots = (
            None
            if os.environ.get(ATTENTION_SPLIT_ENV_VAR, "1") == "0"
            else PREFILL_SLOTS
        )
        self._defer_logits = (
            os.environ.get(DEFER_LOGITS_ENV_VAR, "0") == "1"
        )
        self._log_shapes = os.environ.get("JAX_VLLM_LOG_SHAPES") == "1"
        self._observed_shapes: set[Any] = set()
        self._step_telemetry = os.environ.get("JAX_VLLM_STEP_TELEMETRY") == "1"
        self._step_telemetry_index = 0
        self._roctx = Roctx.from_environment()
        self._roctx_step_index = 0
        self._roctx_selected_regions = (
            os.environ.get(ROCTX_SELECTED_REGIONS_ENV_VAR) == "1"
        )
        self._roctx_expected_tokens = int(
            os.environ.get(ROCTX_EXPECTED_TOKENS_ENV_VAR, "0")
        )
        self._roctx_profile_rep = int(os.environ.get(ROCTX_PROFILE_REP_ENV_VAR, "1"))
        if self._roctx_selected_regions and self._roctx is None:
            raise ValueError(
                f"{ROCTX_SELECTED_REGIONS_ENV_VAR}=1 requires JAX_VLLM_ROCTX=1"
            )
        if self._roctx_selected_regions and self._roctx_expected_tokens <= 0:
            raise ValueError(
                f"{ROCTX_EXPECTED_TOKENS_ENV_VAR} must be positive when "
                f"{ROCTX_SELECTED_REGIONS_ENV_VAR}=1"
            )
        if self._roctx_profile_rep <= 0:
            raise ValueError(f"{ROCTX_PROFILE_REP_ENV_VAR} must be positive")
        self._roctx_profile_started = False
        self._roctx_profile_done = False
        self._roctx_profile_tokens = 0
        self._roctx_profile_range_id: int | None = None
        self._roctx_profile_lock = Lock()
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
        # XPlane's stop condition needs completed steps, so it still cannot run
        # under async. The step ledger can: everything it records is host-side
        # data known before submission, and its timing is reported as `submit`
        # rather than `execute_blocking` on that path. Blocking the ledger under
        # async would leave the arm we actually measure unable to report its own
        # shapes.
        if self._async_scheduling and self._xplane_dir:
            raise ValueError(
                "async scheduling cannot be combined with XPlane tracing"
            )

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
        import jax.numpy as jnp

        from maxtext.inference.kv_execution.bucketing import StepShapePlanner

        from .kv_spec import KV_LAYOUT_NHD, KV_LAYOUT_X_VECTORIZED, build_layout
        from .mesh import assert_kv_heads_shardable, kv_head_shards, pool_replicas

        num_layers, num_kv_heads, head_dim = self.model_geometry()
        assert_kv_heads_shardable(self.maxtext_config, self.mesh, num_kv_heads)

        asm_decode = os.environ.get(ASM_DECODE_ENV_VAR, "0") != "0"
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
            kv_layout=KV_LAYOUT_X_VECTORIZED if asm_decode else KV_LAYOUT_NHD,
        )
        # Columns of the dense page map the ASM kernel takes, from the longest
        # context the model will admit rather than from any live batch, so it is
        # one width for the whole run and adds no compiled shapes. None keeps
        # the host out of building a table nothing reads.
        self._block_table_blocks = (
            -(-int(self.model_config.max_model_len) // self.tokens_per_page)
            if asm_decode
            else None
        )
        if asm_decode:
            logger.info(
                "ASM decode on: x-vectorized KV pool, dense page map %d blocks wide. "
                "%s is ignored for this layout -- the direct_i32 append fast path "
                "writes NHD only.",
                self._block_table_blocks,
                "JAX_AITER_APPEND_KV_DIRECT_I32",
            )

        self.pool = pool_module.allocate(self.layout, self.mesh, self.maxtext_config)
        self.scrub = pool_module.make_scrub_fn(self.pool)
        # Compile every scrub rung now rather than during a measured replay.
        warm_scrub = getattr(self.scrub, "warm", None)
        if warm_scrub is not None:
            warm_scrub(num_blocks)
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
        token_bucket_subdivisions = int(
            os.environ.get(TOKEN_BUCKET_SUBDIVISIONS_ENV_VAR, "2")
        )
        if token_bucket_subdivisions > 8:
            raise ValueError(
                f"{TOKEN_BUCKET_SUBDIVISIONS_ENV_VAR} must be between 1 and 8, "
                f"got {token_bucket_subdivisions}"
            )
        self.planner.token_rungs = _densify_token_rungs(
            self.planner.token_rungs, token_bucket_subdivisions
        )
        logger.info("Serving token buckets: %s.", self.planner.token_rungs)
        with self.mesh:
            self._chained_tokens = jnp.zeros(
                (self._max_request_slots + 1,) if self._async_scheduling else (0,),
                dtype=jnp.int32,
            )
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
        use_async = self._async_scheduling
        defer_logits = self._defer_logits
        dense_table_width = self._block_table_blocks

        @functools.partial(
            jax.jit,
            static_argnames=(
                "num_tokens",
                "num_requests",
                "num_pages",
                "max_seqlen_q",
                "max_seqlen_k",
                "is_decode",
                "num_prefill_slots",
                "all_greedy",
            ),
            donate_argnames=(
                ("kv_caches", "chained_tokens")
                if use_async
                else ("kv_caches",)
            ),
        )
        def step(
            state_leaves,
            kv_caches,
            packed_ints,
            packed_floats,
            chained_tokens,
            rng,
            *,
            num_tokens: int,
            num_requests: int,
            num_pages: int,
            max_seqlen_q: int,
            max_seqlen_k: int,
            is_decode: bool,
            num_prefill_slots: int | None,
            all_greedy: bool,
        ):
            operands = _unpack_step_operands(
                packed_ints,
                packed_floats,
                num_tokens=num_tokens,
                num_requests=num_requests,
                num_pages=num_pages,
                block_table_blocks=dense_table_width,
                include_chaining=use_async,
            )
            (
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
            ) = operands[:11]
            operand_offset = 11
            if dense_table_width is not None:
                block_tables, context_lens = operands[
                    operand_offset : operand_offset + 2
                ]
                operand_offset += 2
            else:
                block_tables = None
                context_lens = None
            if use_async:
                chained_token_sources, request_token_slots = operands[
                    operand_offset:
                ]
                safe_sources = jnp.maximum(chained_token_sources, 0)
                input_ids = jnp.where(
                    chained_token_sources >= 0,
                    chained_tokens[safe_sources],
                    input_ids,
                )
            model = nnx.merge(graphdef, jax.tree_util.tree_unflatten(treedef, state_leaves))
            metadata = paged_plan.PagedPlanWithPositions(
                slot_mapping=slot_mapping,
                kv_indptr=kv_indptr,
                kv_page_indices=kv_page_indices,
                kv_last_page_lens=kv_last_page_lens,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                # Static, and it has to travel with the shape: without it the
                # plan rebuilt in here is not a split plan, and the whole batch
                # goes to the prefill kernel with every decode row declared
                # zero-length -- which is fast and silently wrong.
                num_prefill_requests=num_prefill_slots,
                num_prefill_tokens=(
                    None
                    if num_prefill_slots is None
                    else num_tokens - (num_requests - num_prefill_slots)
                ),
                is_decode=is_decode,
                block_tables=block_tables,
                context_lens=context_lens,
                input_positions=input_positions,
            )

            # Second element is logits on this path, despite the adapter naming
            # it `hidden`; see the module docstring.
            kv_caches, model_out, _aux, _experts = model(
                kv_caches, input_ids, metadata
            )
            if defer_logits:
                # MaxText returned hidden states. Gather the sampled rows first,
                # then project only those to the vocabulary: the head runs on
                # `request_width` rows (136 split / 128 otherwise) instead of
                # every padded token row. `defer_logits` is closed over and
                # constant, so this adds no compiled shape.
                selected = model.compute_logits(
                    jnp.take(model_out, sample_rows, axis=0)
                )
            else:
                selected = jnp.take(model_out, sample_rows, axis=0)
            tokens = sampling.sample(
                rng, selected, (temperature, top_p, top_k), all_greedy
            )
            if use_async:
                chained_tokens = chained_tokens.at[request_token_slots].set(tokens)
            return kv_caches, tokens, chained_tokens

        return step

    def execute_model(self, scheduler_output: Any) -> Any:
        import jax
        import jax.numpy as jnp

        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

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
            view_array_factory=np.asarray,
            num_prefill_requests=(
                count_prefill(requests)
                if shape.num_prefill_slots is not None
                else None
            ),
            block_table_blocks=self._block_table_blocks,
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
        chain_operands = {}
        if self._async_scheduling:
            chain_operands = {
                "chained_token_sources": _chained_token_sources(
                    requests,
                    self.requests,
                    num_tokens=shape.num_tokens,
                    sample_rows=sample_rows,
                ),
                "request_token_slots": _pad(
                    np.asarray(
                        [
                            self.requests[r.request_id].device_token_slot
                            for r in requests
                        ],
                        dtype=np.int32,
                    ),
                    shape.num_requests,
                    self._max_request_slots,
                ),
            }
        packed_ints, packed_floats = _pack_step_operands(
            input_ids,
            positions,
            view,
            sample_rows,
            _pad(spec.temperature, shape.num_requests, 0.0),
            _pad(spec.top_p, shape.num_requests, 1.0),
            _pad(spec.top_k, shape.num_requests, 1),
            **chain_operands,
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
                "range_scope": (
                    "submit" if self._async_scheduling else "completion"
                ),
            }
        if self._roctx is not None:
            roctx_profile_call = bool(
                self._roctx_selected_regions
                and not self._roctx_profile_done
                and profile_metadata["phase"] == "measured"
                and profile_metadata["rep"] == self._roctx_profile_rep
            )
            if roctx_profile_call and not self._roctx_profile_started:
                self._roctx.profiler_resume()
                self._roctx_profile_range_id = self._roctx.start(
                    "JAX_VLLM_PROFILE_REPLAY "
                    f"phase={profile_metadata['phase']} "
                    f"rep={profile_metadata['rep']} "
                    f"expected_true_tokens={self._roctx_expected_tokens}"
                )
                self._roctx_profile_started = True
            roctx_name = "JAX_VLLM_MODEL_CALL " + " ".join(
                f"{key}={value}" for key, value in profile_metadata.items()
            )
            self._roctx.push(roctx_name)
        else:
            roctx_profile_call = False
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
                updated, tokens, chained_tokens = self._step_fn(
                    self._state_leaves,
                    pool_module.as_kv_caches(self.pool),
                    jnp.asarray(packed_ints),
                    jnp.asarray(packed_floats),
                    self._chained_tokens,
                    step_rng,
                    num_tokens=shape.num_tokens,
                    num_requests=shape.num_requests,
                    num_pages=shape.num_pages,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=shape.max_seqlen_k,
                    is_decode=shape.is_decode,
                    num_prefill_slots=shape.num_prefill_slots,
                    all_greedy=spec.all_greedy,
                )
            pool_module.rebind(self.pool, updated)
            self._chained_tokens = chained_tokens
            if not self._async_scheduling:
                sampled = np.asarray(jax.device_get(tokens), dtype=np.int64)
        finally:
            if annotation is not None:
                annotation.__exit__(*sys.exc_info())
            if self._roctx is not None:
                self._roctx.pop()
                if (
                    roctx_profile_call
                    and sys.exc_info()[0] is not None
                    and not self._roctx_profile_done
                ):
                    if self._roctx_profile_range_id is not None:
                        self._roctx.stop(self._roctx_profile_range_id)
                        self._roctx_profile_range_id = None
                    self._roctx.profiler_pause()
                    self._roctx_profile_done = True
        if self._async_scheduling:
            if self._step_telemetry:
                self._emit_step_ledger(
                    requests,
                    prepared=prepared,
                    shape=shape,
                    max_seqlen_q=max_seqlen_q,
                    step_start_ns=step_start_ns,
                    execute_start_ns=execute_start_ns,
                    execute_end_ns=time.monotonic_ns(),
                )
            if self._pending_async_result is not None:
                raise RuntimeError(
                    "sample_tokens was not called after the previous async model step"
                )
            on_complete = None
            if self._roctx is not None:
                on_complete = (
                    lambda metadata=profile_metadata,
                    selected=roctx_profile_call: self._finish_async_roctx_call(
                        metadata,
                        selected,
                    )
                )
            # Capture the RequestTokens objects, not just the ids: vLLM may retire
            # or textually reuse an id before this output is read back, and a
            # dict lookup then raises KeyError or appends this generation's token
            # onto a different request. Object identity, not a copy -- _collect
            # must still append to the live state the chaining path reads.
            self._pending_async_result = (
                tuple(requests),
                tuple(self.requests[r.request_id] for r in requests),
                tokens,
                on_complete,
            )
            return None
        if roctx_profile_call:
            self._account_roctx_profile_tokens(profile_metadata)
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
            self._emit_step_ledger(
                requests,
                prepared=prepared,
                shape=shape,
                max_seqlen_q=max_seqlen_q,
                step_start_ns=step_start_ns,
                execute_start_ns=execute_start_ns,
                execute_end_ns=execute_end_ns,
            )
        return self._model_output(requests, sampled)

    def _emit_step_ledger(
        self,
        requests,
        *,
        prepared: Any,
        shape: Any,
        max_seqlen_q: int,
        step_start_ns: int,
        execute_start_ns: int,
        execute_end_ns: int,
    ) -> None:
        """Log one step's true-versus-padded shape and host timing.

        Emitted on both the synchronous and the asynchronous path, because the
        padding this records is the same work either way. Only the second timing
        interval differs in meaning: synchronously it covers execution and
        readback, asynchronously it covers submission alone, so it is named for
        what it measured rather than reported under one label that would be true
        in only one of the two cases.
        """
        self._step_telemetry_index += 1
        run_phase, run_rep = _run_identity(requests)
        token_split = _scheduled_token_split(requests, self.requests)
        query_lens = np.asarray(prepared.table.query_lens)
        seq_lens = np.asarray(prepared.table.seq_lens)
        second_interval = (
            "submit" if self._async_scheduling else "execute_blocking"
        )
        record = {
            "schema": 2,
            "event": "jax_vllm_step",
            "step_index": self._step_telemetry_index,
            "run_phase": run_phase,
            "run_rep": run_rep,
            "async_scheduling": bool(self._async_scheduling),
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
                second_interval: execute_end_ns - execute_start_ns,
                "timed_total": execute_end_ns - step_start_ns,
            },
        }
        logger.warning(
            "JAX_VLLM_STEP %s",
            json.dumps(record, sort_keys=True, separators=(",", ":")),
        )

    def _finish_async_roctx_call(
        self,
        profile_metadata: dict[str, Any],
        profile_call: bool,
    ) -> None:
        if profile_call:
            self._account_roctx_profile_tokens(profile_metadata)

    def _account_roctx_profile_tokens(
        self,
        profile_metadata: dict[str, Any],
    ) -> None:
        with self._roctx_profile_lock:
            self._roctx_profile_tokens += int(profile_metadata["true_tokens"])
            if self._roctx_profile_tokens > self._roctx_expected_tokens:
                raise RuntimeError(
                    "ROCTX selected-region capture exceeded its expected "
                    f"true-token total: {self._roctx_profile_tokens} > "
                    f"{self._roctx_expected_tokens}"
                )
            if self._roctx_profile_tokens == self._roctx_expected_tokens:
                if self._roctx_profile_range_id is not None:
                    self._roctx.stop(self._roctx_profile_range_id)
                    self._roctx_profile_range_id = None
                self._roctx.profiler_pause()
                self._roctx_profile_done = True

    def sample_tokens(self, grammar_output: Any) -> Any:
        """Start async token readback for the step dispatched by execute_model."""
        if not self._async_scheduling:
            raise RuntimeError("sample_tokens requires async scheduling")
        if grammar_output is not None:
            raise ValueError("structured-output grammar is unsupported")
        if self._pending_async_result is None:
            raise RuntimeError("sample_tokens called without a pending model step")

        from .async_output import JaxAsyncModelRunnerOutput

        requests, states, tokens, on_complete = self._pending_async_result
        self._pending_async_result = None
        return JaxAsyncModelRunnerOutput(
            self,
            requests,
            tokens,
            states=states,
            on_complete=on_complete,
        )

    def _model_output(self, requests, sampled: np.ndarray, states=None) -> Any:
        from vllm.v1.outputs import ModelRunnerOutput

        return ModelRunnerOutput(
            req_ids=[r.request_id for r in requests],
            req_id_to_index={r.request_id: i for i, r in enumerate(requests)},
            sampled_token_ids=self._collect(requests, sampled, states=states),
        )

    # --------------------------------------------------------------- pieces

    def _admit(self, scheduler_output: Any) -> None:
        if self.tracker is None:
            raise RuntimeError("KV cache must be initialized before admitting requests")
        for new in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            if not self._free_token_slots:
                raise ValueError(
                    f"more than {self._max_request_slots} requests are resident"
                )
            self.tracker.admit(new.req_id)
            self.requests[new.req_id] = RequestTokens(
                new.req_id,
                new.prompt_token_ids,
                new.sampling_params,
                self._free_token_slots.pop(),
            )

    def _retire(self, scheduler_output: Any) -> None:
        for finished in getattr(scheduler_output, "finished_req_ids", ()) or ():
            state = self.requests.pop(finished, None)
            if state is not None and state.device_token_slot >= 0:
                self._free_token_slots.append(state.device_token_slot)
            if self.tracker is not None:
                self.tracker.retire(finished)

    def _tokens_for(self, request: RequestPages) -> np.ndarray:
        state = self.requests[request.request_id]
        pending_offset = _pending_token_offset(request, state)
        if pending_offset is None:
            return state.slice(
                request.num_computed_tokens, request.num_scheduled_tokens
            )
        if not self._async_scheduling:
            return state.slice(
                request.num_computed_tokens, request.num_scheduled_tokens
            )
        tokens = np.zeros((request.num_scheduled_tokens,), dtype=np.int32)
        if pending_offset:
            tokens[:pending_offset] = state.slice(
                request.num_computed_tokens, pending_offset
            )
        return tokens

    def _shape_for(self, requests) -> Any:
        query_lens = [r.num_scheduled_tokens for r in requests]
        max_seq_len = max((r.seq_len for r in requests), default=1)
        if all(length == 1 for length in query_lens):
            return self.planner.decode_shape(len(requests), max_seq_len)
        prefilling = count_prefill(requests)
        return self.planner.extend_shape(
            sum(query_lens),
            max_seq_len,
            num_prefill_requests=prefilling,
            num_prefill_tokens=sum(query_lens[:prefilling]),
            prefill_slots=self._prefill_slots,
        )

    def _collect(self, requests, sampled: np.ndarray, states=None) -> list[list[int]]:
        """One sampled token per request, or none while it is still prefilling.

        Under chunked prefill a request is scheduled for several steps before
        its prompt is consumed, and the token predicted from a partial prompt is
        not a continuation of anything. vLLM reads an empty list as "no output
        this step", which is exactly the case.
        """
        out: list[list[int]] = []
        if states is None:
            states = [self.requests[request.request_id] for request in requests]
        for row, (request, state) in enumerate(zip(requests, states, strict=True)):
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
