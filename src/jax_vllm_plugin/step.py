# SPDX-License-Identifier: Apache-2.0
"""The per-step pipeline, in the order the recycled-page guarantee requires.

MaxText's native runtime fixes this order in `PagedRuntime.prepare_step`:
reserve, scrub, confirm, build, run. Under vLLM the reserve step belongs to
vLLM, but the rest of the order still has to hold, and it is the *order* that
carries the guarantee rather than any single call.

Scrub before the page table is built, not after: a table naming an unscrubbed
page is refused, so building first and scrubbing later would either raise or,
worse, be reordered by a well-meaning future edit into a step that reads another
request's KV.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import numpy as np

from maxtext.inference.kv_common import KvPageTableV1
from maxtext.inference.kv_execution.step_inputs import RequestSlice, StepInputs, build_step_inputs
from maxtext.inference.kv_execution.step_view import StepView, build_step_view

from .kv_reuse import PageReuseTracker
from .page_table import RequestPages, build_page_table

# Zeroes the named pool pages on device. Separate from the tracker because the
# tracker is host bookkeeping and must stay testable without a GPU.
ScrubFn = Callable[[np.ndarray], None]


class PreparedStep(NamedTuple):
    view: StepView
    inputs: StepInputs
    table: KvPageTableV1
    requests: tuple[RequestPages, ...]
    scrubbed: np.ndarray


def prepare_step(
    requests: Sequence[RequestPages],
    *,
    tracker: PageReuseTracker,
    scrub: ScrubFn,
    shape: Any,
    tokens_per_page: int,
    token_ids: Callable[[RequestPages], np.ndarray],
    padding_page_id: int = 0,
    view_array_factory: Callable[..., Any] | None = None,
    num_prefill_requests: int | None = None,
    block_table_blocks: int | None = None,
) -> PreparedStep:
    """Run one step's host-side preparation.

    Args:
      requests: this step's per-request pages, in page-table row order.
      tracker: recycled-page bookkeeping.
      scrub: zeroes pool pages on device.
      shape: the bucketed `StepShape` to pad to.
      tokens_per_page: pool page size, equal to vLLM's block_size.
      token_ids: yields exactly the tokens this step feeds for a request, whose
        length must equal its scheduled token count.
      padding_page_id: pool page reserved as a padding target.
      num_prefill_requests: how many leading requests are chunk-prefilling, when
        `shape` reserves prefill slots. The view and the token packing both need
        it to place the decode half at the static split offset.
      block_table_blocks: columns of the dense page map AITER's ASM decode
        takes, or None to skip building it. Must come from the bucket rather
        than the live batch, or it introduces a new compiled shape per context
        length.
    """
    pending = tracker.observe(requests)
    if pending.size:
        scrub(pending)
        tracker.confirm_scrubbed(pending)

    table = build_page_table(requests, tokens_per_page=tokens_per_page)

    # The refusal that makes the guarantee structural rather than remembered.
    tracker.assert_clean(table.flat_page_indices())

    view_kwargs = {"padding_page_id": padding_page_id}
    if view_array_factory is not None:
        view_kwargs["array_factory"] = view_array_factory
    if num_prefill_requests is not None:
        view_kwargs["num_prefill_requests"] = num_prefill_requests
    if block_table_blocks is not None:
        view_kwargs["block_table_blocks"] = block_table_blocks
    view = build_step_view(table, shape, tokens_per_page, **view_kwargs)

    slices = []
    for request in requests:
        tokens = np.asarray(token_ids(request))
        if tokens.shape[0] != request.num_scheduled_tokens:
            raise ValueError(
                f"request {request.request_id!r} scheduled "
                f"{request.num_scheduled_tokens} tokens but token_ids yielded "
                f"{tokens.shape[0]}"
            )
        slices.append(
            RequestSlice(
                tokens=tokens,
                start=request.num_computed_tokens,
                query_len=request.num_scheduled_tokens,
            )
        )

    inputs = build_step_inputs(
        slices,
        shape,
        is_decode=shape.is_decode,
        num_prefill_requests=num_prefill_requests,
    )

    # Only after the table is accepted, so a step that raised leaves no trace
    # suggesting its pages were written.
    tracker.note_written(requests)

    return PreparedStep(
        view=view,
        inputs=inputs,
        table=table,
        requests=tuple(requests),
        scrubbed=pending,
    )
