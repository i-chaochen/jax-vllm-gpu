# SPDX-License-Identifier: Apache-2.0
"""Per-request state carried across steps.

vLLM sends a request's full block table once, then sends deltas. A runner has to
accumulate them, because the page table for a step needs the whole context, not
just the newly appended tail.

Three cases have to be distinguished and the difference is not cosmetic:

* a **new** request arrives with its complete block table,
* a **continuing** request arrives with only the blocks appended this step,
  which are appended to what we already hold,
* a **preempted** request is named in `preempted_req_ids`: it has given up its
  blocks, so its state is dropped, and vLLM sends it again later as a new
  request with a fresh block table.

Keeping a preempted request's blocks silently produces a page table naming pages
the request no longer owns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .page_table import RequestPages


def retired_request_ids(scheduler_output: Any) -> set[str]:
    """Requests whose state ends with this scheduler output: finished or preempted."""
    finished = getattr(scheduler_output, "finished_req_ids", None) or ()
    preempted = getattr(scheduler_output, "preempted_req_ids", None) or ()
    return set(finished) | set(preempted)


@dataclass
class RequestState:
    request_id: str
    block_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0


def _single_group(block_ids: Any, request_id: str) -> list[int]:
    """Take the one KV cache group's block list.

    vLLM carries `tuple[list[int], ...]`, one entry per group. Hybrid models have
    several; this platform serves uniform full attention, so more than one group
    means the caller built a spec we cannot honour and should say so loudly.
    """
    if not isinstance(block_ids, (tuple, list)):
        raise TypeError(
            f"request {request_id!r} block_ids should be a tuple of per-group "
            f"lists, got {type(block_ids).__name__}"
        )
    if len(block_ids) != 1:
        raise ValueError(
            f"request {request_id!r} has {len(block_ids)} KV cache groups; this "
            "platform supports exactly one (uniform full attention)"
        )
    return list(block_ids[0])


class BatchState:
    """Accumulates the block tables and computed-token counts vLLM sends."""

    def __init__(self) -> None:
        self._requests: dict[str, RequestState] = {}

    def __len__(self) -> int:
        return len(self._requests)

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._requests

    def get(self, request_id: str) -> RequestState:
        return self._requests[request_id]

    def apply(self, scheduler_output: Any) -> None:
        """Fold one `SchedulerOutput` into the accumulated state."""
        # Retire before admitting: a finished request's id can be reused, and a
        # preempted request can come back, in the same scheduler output.
        for request_id in retired_request_ids(scheduler_output):
            self._requests.pop(request_id, None)

        for new in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            if new.req_id in self._requests:
                raise ValueError(f"request {new.req_id!r} is already active")
            self._requests[new.req_id] = RequestState(
                request_id=new.req_id,
                block_ids=_single_group(new.block_ids, new.req_id),
                num_computed_tokens=new.num_computed_tokens,
            )

        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is not None:
            self._apply_cached(cached)

    def _apply_cached(self, cached: Any) -> None:
        req_ids = list(cached.req_ids)
        new_block_ids = list(cached.new_block_ids)
        num_computed = list(cached.num_computed_tokens)

        if not (len(req_ids) == len(new_block_ids) == len(num_computed)):
            raise ValueError(
                "CachedRequestData is ragged: "
                f"{len(req_ids)} ids, {len(new_block_ids)} block entries, "
                f"{len(num_computed)} computed counts"
            )

        for request_id, blocks, computed in zip(req_ids, new_block_ids, num_computed):
            state = self._requests.get(request_id)
            if state is None:
                raise KeyError(
                    f"request {request_id!r} was scheduled as cached but was "
                    "never introduced as new"
                )
            state.num_computed_tokens = computed
            if blocks is None:
                continue
            state.block_ids.extend(_single_group(blocks, request_id))

    def pages_for_step(self, scheduler_output: Any) -> list[RequestPages]:
        """Assemble this step's per-request pages, in a stable order.

        Ordering follows `num_scheduled_tokens`, which is what vLLM iterates, so
        the rows of the page table and of the sampled output agree.
        """
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
        pages: list[RequestPages] = []
        for request_id, num_tokens in scheduled.items():
            state = self._requests.get(request_id)
            if state is None:
                raise KeyError(
                    f"request {request_id!r} has scheduled tokens but no state"
                )
            pages.append(
                RequestPages(
                    request_id=request_id,
                    block_ids=state.block_ids,
                    num_computed_tokens=state.num_computed_tokens,
                    num_scheduled_tokens=num_tokens,
                )
            )
        return pages
