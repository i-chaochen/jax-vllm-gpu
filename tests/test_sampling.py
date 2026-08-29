# SPDX-License-Identifier: Apache-2.0
"""Sampling, which runs on device but is asserted on the host."""

from __future__ import annotations

import numpy as np
import pytest

from jax_vllm_plugin import sampling

jax = pytest.importorskip("jax", reason="sampling runs in JAX")
jnp = pytest.importorskip("jax.numpy")


class Params:
    """Enough of vLLM's `SamplingParams` for `from_params`."""

    def __init__(self, temperature=0.0, top_p=1.0, top_k=0):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k


def test_a_zero_temperature_batch_is_recognised_as_greedy():
    """Worth a test rather than a comment: `all_greedy` selects a materially
    cheaper traced program, so getting it wrong is a silent slowdown."""
    spec = sampling.from_params([Params(), Params()], vocab_size=32)

    assert spec.all_greedy


def test_one_stochastic_request_makes_the_whole_batch_stochastic():
    spec = sampling.from_params([Params(), Params(temperature=0.8)], vocab_size=32)

    assert not spec.all_greedy


def test_a_missing_sampling_params_reads_as_greedy():
    """A pooling request carries none, and it will not be sampled from anyway."""
    spec = sampling.from_params([None], vocab_size=32)

    assert spec.all_greedy
    assert spec.top_k[0] == 32


def test_top_k_of_zero_means_the_whole_vocabulary():
    """vLLM spells 'no truncation' as 0 in some paths and -1 in others."""
    spec = sampling.from_params([Params(top_k=0), Params(top_k=-1)], vocab_size=17)

    assert list(spec.top_k) == [17, 17]


def test_greedy_sampling_picks_the_largest_logit():
    logits = jnp.asarray([[0.0, 5.0, 1.0], [7.0, 0.0, 2.0]])
    tokens = sampling.sample(jax.random.key(0), logits, None, all_greedy=True)

    assert list(np.asarray(tokens)) == [1, 0]


def test_a_zero_temperature_row_stays_greedy_in_a_mixed_batch():
    """The mixed path divides by temperature, and a zero there would produce
    infinities. The row must come back exactly greedy, not merely close."""
    logits = jnp.asarray([[0.0, 5.0, 1.0], [0.0, 1.0, 2.0]])
    spec = sampling.from_params([Params(), Params(temperature=1.0)], vocab_size=3)
    arrays = (
        jnp.asarray(spec.temperature),
        jnp.asarray(spec.top_p),
        jnp.asarray(spec.top_k),
    )

    tokens = np.asarray(sampling.sample(jax.random.key(0), logits, arrays, False))

    assert tokens[0] == 1


def test_top_k_of_one_forces_the_argmax_however_hot_the_temperature():
    """A high temperature would otherwise spread the draw across the vocabulary,
    so this isolates the mask rather than the sampler."""
    logits = jnp.asarray([[0.0, 5.0, 4.9, 4.8]])
    spec = sampling.from_params([Params(temperature=100.0, top_k=1)], vocab_size=4)
    arrays = (
        jnp.asarray(spec.temperature),
        jnp.asarray(spec.top_p),
        jnp.asarray(spec.top_k),
    )

    drawn = {
        int(sampling.sample(jax.random.key(seed), logits, arrays, False)[0])
        for seed in range(16)
    }

    assert drawn == {1}


def test_a_tiny_top_p_still_leaves_one_token_to_draw():
    """The top token alone can already exceed top_p. Masking everything would
    leave a row of -inf and a NaN draw, so column zero is pinned."""
    logits = jnp.asarray([[0.0, 10.0, 0.0, 0.0]])
    spec = sampling.from_params([Params(temperature=1.0, top_p=1e-6)], vocab_size=4)
    arrays = (
        jnp.asarray(spec.temperature),
        jnp.asarray(spec.top_p),
        jnp.asarray(spec.top_k),
    )

    drawn = {
        int(sampling.sample(jax.random.key(seed), logits, arrays, False)[0])
        for seed in range(8)
    }

    assert drawn == {1}
