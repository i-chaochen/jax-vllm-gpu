# SPDX-License-Identifier: Apache-2.0
"""Recycled-page tracking against vLLM's block lifecycle.

MaxText's native runtime gets this from its own allocator: a page is marked
dirty when freed, and a page table naming a dirty page is refused until the page
has been zeroed. Under vLLM the allocator is gone, so the obligation has to be
reconstructed from what a step actually tells us.

vLLM recycles blocks without zeroing them -- `get_new_blocks` resets a block's
hash and nothing else -- and it only populates `new_block_ids_to_zero` for Mamba
or mixed-precision caches, so a uniform-precision attention deployment gets no
zeroing at all. Upstream is safe because its kernels never read past `seq_len`.

The trigger has to be chosen carefully, because scrubbing too much is as wrong
as scrubbing too little:

* **Read region**, `block_ids[:num_computed_tokens // page]`, holds KV the
  request is meant to read. A prefix-cache hit lands here. Zeroing it destroys
  both the hit and correctness. Never scrub it.
* **Write region**, the remainder, is scratch. Recycled pages can only appear
  here, because cache hits reach a request through `touch()` rather than
  `get_new_blocks()`.

So the obligation is: pages in the write region that we have previously written
for a *different* request. A page the same request already owns is excluded --
re-zeroing a partially filled page would wipe its own valid prefix.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .page_table import RequestPages


class PageReuseTracker:
    """Tracks which pool pages hold a previous occupant's KV.

    Deliberately shaped like MaxText's `KvControlPlane` scrub contract:
    `pending_scrub` is idempotent and only `confirm_scrubbed` discharges the
    obligation, so a caller that forgets to zero on device cannot accidentally
    clear the record by reading it.
    """

    _NO_OWNER = -1

    def __init__(self, num_pages: int, tokens_per_page: int) -> None:
        if num_pages <= 0:
            raise ValueError(f"num_pages must be positive, got {num_pages}")
        if tokens_per_page <= 0:
            raise ValueError(f"tokens_per_page must be positive, got {tokens_per_page}")

        self.num_pages = num_pages
        self.tokens_per_page = tokens_per_page

        # Pages start clean because the pool is allocated with jnp.zeros. This
        # is a real dependency on allocate_pool, not an assumption about the
        # allocator handing out fresh memory.
        self._written = np.zeros(num_pages, dtype=bool)
        self._owner = np.full(num_pages, self._NO_OWNER, dtype=np.int64)
        self._request_slots: dict[str, int] = {}
        self._next_owner = 0
        self._pending: set[int] = set()

    def admit(self, request_id: str) -> int:
        """Assign a fresh owner generation to a newly admitted request."""
        if request_id in self._request_slots:
            raise ValueError(f"request {request_id!r} is already active")
        owner = self._next_owner
        self._next_owner += 1
        self._request_slots[request_id] = owner
        return owner

    def retire(self, request_id: str) -> None:
        """Forget the active name without erasing ownership left in pages."""
        self._request_slots.pop(request_id, None)

    def _slot(self, request_id: str) -> int:
        slot = self._request_slots.get(request_id)
        if slot is None:
            # Direct control-plane users may not have an explicit admission
            # event. The model runner does, and calls `admit` itself.
            slot = self.admit(request_id)
        return slot

    def write_region(self, request: RequestPages) -> list[int]:
        """Pages this step may write into, excluding the already-valid prefix."""
        first = request.num_computed_tokens // self.tokens_per_page
        needed = request.pages_needed(self.tokens_per_page)
        return list(request.block_ids[first:needed])

    def observe(self, requests: Sequence[RequestPages]) -> np.ndarray:
        """Record the scrub obligation implied by this step's block tables.

        Returns the pages needing zeroing, which is also accumulated so a
        caller that observes twice before scrubbing does not lose any.
        """
        for request in requests:
            slot = self._slot(request.request_id)
            for page in self.write_region(request):
                if page < 0 or page >= self.num_pages:
                    raise ValueError(
                        f"page {page} out of range for a pool of {self.num_pages}"
                    )
                if self._written[page] and self._owner[page] != slot:
                    self._pending.add(int(page))
        return self.pending_scrub()

    def pending_scrub(self) -> np.ndarray:
        """Pages awaiting a device-side zero. Reading does not discharge."""
        if not self._pending:
            return np.empty((0,), dtype=np.int32)
        return np.asarray(sorted(self._pending), dtype=np.int32)

    def confirm_scrubbed(self, page_ids: Sequence[int] | np.ndarray) -> None:
        """Discharge the obligation for pages that have been zeroed on device."""
        for page in np.asarray(page_ids, dtype=np.int64).reshape(-1).tolist():
            self._pending.discard(int(page))
            self._written[page] = False
            self._owner[page] = self._NO_OWNER

    def note_written(self, requests: Sequence[RequestPages]) -> None:
        """Record that this step wrote KV into each request's write region."""
        for request in requests:
            slot = self._slot(request.request_id)
            for page in self.write_region(request):
                self._written[page] = True
                self._owner[page] = slot

    def assert_clean(self, page_ids: Sequence[int] | np.ndarray) -> None:
        """Refuse a page table that still names an unscrubbed page.

        The analogue of MaxText's `build_page_table` raising `DirtyPageError`.
        Making it an error rather than a convention is the point: the failure it
        prevents -- one request reading another's KV -- is silent.
        """
        pages = np.asarray(page_ids, dtype=np.int64).reshape(-1)
        unscrubbed = [int(p) for p in pages if int(p) in self._pending]
        if unscrubbed:
            raise DirtyPageError(
                f"pages {sorted(unscrubbed)[:8]} still hold a previous "
                "request's KV; scrub them and call confirm_scrubbed first"
            )

    @property
    def num_pending(self) -> int:
        return len(self._pending)

    @property
    def num_active_requests(self) -> int:
        return len(self._request_slots)


class DirtyPageError(RuntimeError):
    """A page table named a page that has not been scrubbed since reuse."""
