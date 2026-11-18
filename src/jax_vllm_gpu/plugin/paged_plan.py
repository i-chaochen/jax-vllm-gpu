# SPDX-License-Identifier: Apache-2.0
"""The metadata object handed to MaxText's model, on the device side.

Two consumers read it and they want different things, which is why this is a
type rather than a dict:

* `Attention.forward_serve_gpu_paged` passes it to `build_plan`, whose first
  branch is `isinstance(metadata, PagedPlan)` -- so subclassing, rather than
  duck-typing, is what makes the conversion a no-op. The alternative is the
  vLLM-shaped route through `plan_from_vllm`, which would rebuild on device
  arrays that `StepView` has already assembled and padded on the host.
* MaxText's vLLM adapter reads `metadata.input_positions` to drive RoPE, and
  `PagedPlan` has no such field.

Padding is the reason the `StepView` route is preferred over the vLLM-shaped
one. `StepView` pads `slot_mapping` with `-1`, which `append_kv` documents as
"skip this token"; a plan derived from padded `block_tables` instead resolves
those rows to real page slots and writes junk KV into them.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from maxtext.layers.gpu_paged_attention import PagedPlan


@dataclasses.dataclass(frozen=True)
class PagedPlanWithPositions(PagedPlan):
    """A `PagedPlan` that also carries the positions the adapter needs."""

    # int32 [num_tokens], absolute position of each token within its own
    # sequence. Absolute rather than step-relative, which is what makes RoPE
    # agree between a request's prefill and its decode steps.
    input_positions: Any = None


def from_step_view(view: Any, input_positions: Any) -> PagedPlanWithPositions:
    """Attach positions to the plan a `StepView` already describes."""
    plan = view.to_paged_plan()
    return PagedPlanWithPositions(
        slot_mapping=plan.slot_mapping,
        kv_indptr=plan.kv_indptr,
        kv_page_indices=plan.kv_page_indices,
        kv_last_page_lens=plan.kv_last_page_lens,
        cu_seqlens_q=plan.cu_seqlens_q,
        max_seqlen_q=plan.max_seqlen_q,
        max_seqlen_k=plan.max_seqlen_k,
        is_decode=plan.is_decode,
        input_positions=input_positions,
    )
