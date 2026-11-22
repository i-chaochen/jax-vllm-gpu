# SPDX-License-Identifier: Apache-2.0
"""Token sampling, in JAX.

vLLM's own sampler is torch, and torch here is a CPU-only build, so pulling
logits back to the host to sample would add a device-to-host round trip to every
decode step -- the one place in the loop where latency is entirely visible.

The transforms are applied per row with everything vectorised, so a batch mixing
greedy and stochastic requests takes one traced path rather than branching.
`temperature <= 0` selects greedy for that row, which is how vLLM spells it too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Below this a temperature is treated as a request for greedy decoding. Matches
# vLLM's own epsilon so a request that is greedy there is greedy here.
_EPS = 1e-5


@dataclass(frozen=True)
class SamplingSpec:
    """Per-request sampling parameters, as batched host arrays.

    One row per request, in page-table row order, so a row of this lines up with
    the same row of the logits.
    """

    temperature: np.ndarray
    top_p: np.ndarray
    top_k: np.ndarray

    @property
    def all_greedy(self) -> bool:
        """True when no row needs the stochastic path.

        Worth testing on the host: it selects a materially cheaper traced
        program, and the common serving case of an all-greedy batch would
        otherwise pay for sorting the vocabulary.
        """
        return bool(np.all(self.temperature <= _EPS))


def from_params(params_by_request, vocab_size: int) -> SamplingSpec:
    """Build a spec from vLLM `SamplingParams`, one per request in batch order.

    A missing `SamplingParams` -- which is what a pooling request carries -- is
    read as greedy rather than rejected, since it will not be sampled from.
    """
    temperature, top_p, top_k = [], [], []
    for params in params_by_request:
        if params is None:
            temperature.append(0.0)
            top_p.append(1.0)
            top_k.append(vocab_size)
            continue
        temperature.append(float(getattr(params, "temperature", 0.0) or 0.0))
        top_p.append(float(getattr(params, "top_p", 1.0) or 1.0))
        # vLLM spells "no truncation" as 0 or -1 depending on the code path;
        # both mean the whole vocabulary.
        k = int(getattr(params, "top_k", 0) or 0)
        top_k.append(vocab_size if k <= 0 else min(k, vocab_size))

    return SamplingSpec(
        temperature=np.asarray(temperature, dtype=np.float32),
        top_p=np.asarray(top_p, dtype=np.float32),
        top_k=np.asarray(top_k, dtype=np.int32),
    )


def _apply_top_k(logits, top_k):
    """Mask all but each row's `top_k` highest logits."""
    import jax.numpy as jnp

    vocab = logits.shape[-1]
    # Descending sort once, then compare each position's rank against the row's
    # k. Cheaper than a per-row `top_k` call and it keeps k traced rather than
    # static, which is what lets one trace serve a batch of mixed k.
    ordered = jnp.sort(logits, axis=-1)[:, ::-1]
    index = jnp.clip(top_k, 1, vocab) - 1
    threshold = jnp.take_along_axis(ordered, index[:, None], axis=-1)
    return jnp.where(logits < threshold, -jnp.inf, logits)


def _apply_top_p(logits, top_p):
    """Mask the tail beyond each row's cumulative probability `top_p`."""
    import jax.nn
    import jax.numpy as jnp

    order = jnp.argsort(logits, axis=-1)[:, ::-1]
    ordered = jnp.take_along_axis(logits, order, axis=-1)
    probability = jax.nn.softmax(ordered, axis=-1)

    # Keep everything whose *exclusive* cumulative mass is still below the
    # threshold, which is the same as keeping the first token that crosses it.
    # Pinning column zero guarantees a survivor even when the top token alone
    # already exceeds top_p.
    keep = jnp.cumsum(probability, axis=-1) - probability < top_p[:, None]
    keep = keep.at[:, 0].set(True)

    masked = jnp.where(keep, ordered, -jnp.inf)
    # Undo the sort so the vocabulary axis means what it did on the way in.
    return jnp.take_along_axis(masked, jnp.argsort(order, axis=-1), axis=-1)


def sample(rng, logits, spec_arrays, all_greedy: bool):
    """Sample one token per row of `logits`.

    Args:
      rng: a JAX PRNG key, split by the caller so each step differs.
      logits: `[num_rows, vocab]`.
      spec_arrays: `(temperature, top_p, top_k)` as device arrays.
      all_greedy: host-side, so the stochastic path can be traced away entirely.

    Returns:
      int32 `[num_rows]`.
    """
    import jax.numpy as jnp
    import jax.random

    logits = logits.astype(jnp.float32)
    greedy = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    if all_greedy:
        return greedy

    temperature, top_p, top_k = spec_arrays
    # Guard the divide before the select rather than after: a zero temperature
    # would produce inf, and the NaNs that follow from it in the categorical
    # draw are not confined to the rows the select later discards.
    safe = jnp.where(temperature > _EPS, temperature, 1.0)
    scaled = logits / safe[:, None]
    scaled = _apply_top_k(scaled, top_k)
    scaled = _apply_top_p(scaled, top_p)

    drawn = jax.random.categorical(rng, scaled, axis=-1).astype(jnp.int32)
    return jnp.where(temperature > _EPS, drawn, greedy)
