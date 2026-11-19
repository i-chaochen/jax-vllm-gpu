"""MaxText's Attention layer, serving through this repository's paged attention.

`models/adapters/maxtext.install()` binds `Attention.forward_serve_gpu_paged`
to `jax_vllm_gpu.models.adapters.maxtext.forward_serve_gpu_paged`, so these
tests exercise this repository's implementation inside MaxText's layer: the
dense-path parity, the kernel arguments of both branches, and the dry run.

Copyright 2026 Advanced Micro Devices, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
import unittest
from unittest import mock

from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.common.gcloud_stub import is_decoupled
from maxtext.configs import pyconfig
from maxtext.layers.attentions import Attention
from maxtext.utils import maxtext_utils

from jax_vllm_gpu.kv.common import KvPageTableV1
from jax_vllm_gpu.models.adapters import maxtext as maxtext_adapter
from jax_vllm_gpu.ops import paged_attention as gpa
from maxtext_test_config import get_test_config_path


class PagedAttentionLayerTest(parameterized.TestCase):
  """`Attention` with `attention='gpu_paged'`, bound to this repository."""

  config_arguments = {
      "per_device_batch_size": 1.0,
      "run_name": "test",
      "enable_checkpointing": False,
      "max_target_length": 512,
  }

  def setUp(self):
    super().setUp()
    maxtext_adapter.install()
    if not is_decoupled():
      jax.config.update("jax_remove_size_one_mesh_axis_from_type", True)
    self.cfg = pyconfig.initialize([sys.argv[0], get_test_config_path()], **self.config_arguments)
    self.rng = jax.random.PRNGKey(0)
    self.nnx_rng = nnx.Rngs(params=0, dropout=jax.random.PRNGKey(42))
    self.mesh = Mesh(maxtext_utils.create_device_mesh(self.cfg), self.cfg.mesh_axes)
    self.global_batch_size = self.cfg.global_batch_size_to_train_on

  def _gpu_paged_config(self, seq_len, **overrides):
    arguments = self.config_arguments.copy()
    arguments["max_target_length"] = seq_len
    arguments["max_prefill_predict_length"] = seq_len
    arguments["scan_layers"] = False
    # Pool capacity has no safe default, so `gpu_paged` requires it. Enough here
    # for one request of this length per batch row.
    arguments["paged_num_blocks"] = 8 * (seq_len // 16 + 1)
    arguments.update(overrides)
    return pyconfig.initialize([sys.argv[0], get_test_config_path()], **arguments)

  def _build_attention(self, config, seq_len, model_mode):
    dummy = jnp.ones((self.global_batch_size, seq_len, config.base_emb_dim))
    return Attention(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=seq_len,
        max_prefill_predict_length=seq_len,
        inputs_q_shape=dummy.shape,
        inputs_kv_shape=dummy.shape,
        mesh=self.mesh,
        attention_kernel="dot_product",
        dtype=config.dtype,
        dropout_rate=config.dropout_rate,
        model_mode=model_mode,
        rngs=self.nnx_rng,
    )

  def _page_table(self, batch, seq_len, tokens_per_page):
    pages_per_request = seq_len // tokens_per_page
    # Page 0 is the padding sentinel and is never handed out.
    return KvPageTableV1(
        page_ids=[[1 + r * pages_per_request + p for p in range(pages_per_request)] for r in range(batch)],
        seq_lens=np.full((batch,), seq_len, np.int32),
        query_lens=np.full((batch,), seq_len, np.int32),
        write_positions=np.tile(np.arange(seq_len, dtype=np.int32), batch),
        request_order=np.arange(batch, dtype=np.int32),
    )

  def test_the_layer_runs_this_repositorys_implementation(self):
    self.assertTrue(maxtext_adapter.installed())

  @pytest.mark.gpu_only
  def test_gpu_paged_matches_the_dense_path(self):
    """`gpu_paged` prefill must agree with dense attention over the same tokens.

    Both sides see identical weights and identical inputs, and a full prefill of
    a whole sequence is exactly causal attention over that sequence, so the two
    are comparable directly. The dense side runs in TRAIN mode specifically so
    neither module allocates a dense KV cache, which keeps the two parameter
    trees identical and lets `nnx.update` copy weights across.
    """
    try:
      from jax_aiter.ffi.registry import standalone_symbol_available  # pylint: disable=import-outside-toplevel
    except ImportError:
      self.skipTest("jax-aiter is not importable; set PYTHONPATH to the jax-aiter checkout")
    for symbol in ("AppendKvJA", "PagedPrefillJA"):
      if not standalone_symbol_available(symbol):
        self.skipTest(f"{symbol} is not built; run 'make -f Makefile.kv ja_kv' in jax-aiter")

    seq_len, tokens_per_page = 128, 16
    dense_cfg = self._gpu_paged_config(seq_len, attention="dot_product")
    paged_cfg = self._gpu_paged_config(seq_len, attention="gpu_paged")

    batch = self.global_batch_size
    lnx = jax.random.normal(self.rng, (batch, seq_len, dense_cfg.base_emb_dim), dtype=dense_cfg.dtype)
    # One segment per request, positions in order: the plain causal case, which
    # is what a full prefill of the sequence computes.
    segment_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch, seq_len))

    dense = self._build_attention(dense_cfg, seq_len, MODEL_MODE_TRAIN)
    dense_out, _ = dense(
        lnx,
        lnx,
        decoder_segment_ids=segment_ids,
        inputs_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    paged = self._build_attention(paged_cfg, seq_len, MODEL_MODE_PREFILL)
    nnx.update(paged, nnx.state(dense))
    self.assertIsNone(paged.KVCache_0, "gpu_paged must not allocate the dense KV cache")

    pages_per_request = seq_len // tokens_per_page
    pool_shape = (1 + batch * pages_per_request, tokens_per_page, paged_cfg.num_kv_heads, paged_cfg.head_dim)
    pools = [jnp.zeros(pool_shape, paged_cfg.dtype), jnp.zeros(pool_shape, paged_cfg.dtype)]

    paged_out, updated = paged(
        lnx,
        lnx,
        decoder_segment_ids=segment_ids,
        inputs_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_PREFILL,
        kv_cache=pools,
        attention_metadata=self._page_table(batch, seq_len, tokens_per_page),
    )

    self.assertEqual(len(updated), 2, "the step must return both pools")
    self.assertEqual(paged_out.shape, dense_out.shape)

    got = np.asarray(paged_out, np.float32)
    want = np.asarray(dense_out, np.float32)
    # One ulp of the output dtype at the scale of the tensor, plus the usual
    # relative term. A purely element-wise relative metric is meaningless here:
    # attention output passes through zero, where one ulp reads as a 100% miss.
    #
    # One ulp is what the two paths actually achieve, so the bound is set to it
    # rather than to a comfortable multiple of it: anything worse than the
    # dtype's own resolution is a real difference. It caught a double-scaling bug
    # where passing the kernel's default 1/sqrt(head_dim) on top of MaxText's
    # already-scaled query produced an error fifty times larger.
    eps = float(jnp.finfo(paged_cfg.dtype).eps)
    atol = eps * max(float(np.abs(want).max()), 1e-6)
    self.assertTrue(
        np.all(np.abs(got - want) <= atol + eps * np.abs(want)),
        f"gpu_paged disagrees with the dense path: max abs error "
        f"{float(np.abs(got - want).max()):.5f}, atol {atol:.5f}",
    )

  def test_gpu_paged_sharded_and_single_device_kernel_arguments_agree(self):
    """Both branches of `forward_serve_gpu_paged` must configure the kernel alike.

    The two branches -- `paged_attention_step` on one device and
    `paged_attention_step_sharded` under `shard_map` -- are the same attention
    and differ only in how the operands are partitioned. Any difference in
    `scale`, `causal` or `backend` between them is a bug by construction.

    This is not a hypothetical. The sharded branch was once written spelling its
    own arguments out, omitted `scale=1.0`, and so let the kernel apply its
    default 1/sqrt(head_dim) on top of a query MaxText had already scaled. Every
    tensor-parallel run then produced a flattened softmax and a wrong token,
    while single-device runs stayed correct.

    The dense-path parity test cannot catch this -- it runs on an unsharded mesh
    and only reaches the single-device branch -- and neither can a comparison of
    the sharded step against the single-device step, because such a test hands
    the same scale to both sides and the error cancels. Asserting on the call
    site closes that gap, so this test mocks the kernels away and inspects the
    arguments rather than the numbers.
    """
    seq_len = 128
    tokens_per_page = 16
    cfg = self._gpu_paged_config(seq_len, attention="gpu_paged")
    paged = self._build_attention(cfg, seq_len, MODEL_MODE_PREFILL)

    batch = self.global_batch_size
    pages_per_request = seq_len // tokens_per_page
    page_table = self._page_table(batch, seq_len, tokens_per_page)
    pool_shape = (1 + batch * pages_per_request, tokens_per_page, cfg.num_kv_heads, cfg.head_dim)
    pools = [jnp.zeros(pool_shape, cfg.dtype), jnp.zeros(pool_shape, cfg.dtype)]
    qkv = jnp.ones((batch, seq_len, cfg.num_query_heads, cfg.head_dim), cfg.dtype)
    kv = jnp.ones((batch, seq_len, cfg.num_kv_heads, cfg.head_dim), cfg.dtype)

    captured = {}

    def record(name):
      def fake(query, *_args, **kwargs):
        # Only the tuning knobs matter here; the operands differ between the two
        # branches by design, since one is global and the other per-shard.
        captured[name] = {k: kwargs[k] for k in ("backend", "scale", "causal") if k in kwargs}
        return query, [pools[0], pools[1]]

      return fake

    with mock.patch.object(gpa, "paged_attention_step", record("single")), mock.patch.object(
        gpa, "paged_attention_step_sharded", record("sharded")
    ), mock.patch.object(gpa, "kv_head_axes", return_value=()):
      paged.forward_serve_gpu_paged(qkv, kv, kv, kv_pools=pools, metadata=page_table)

    with mock.patch.object(gpa, "paged_attention_step", record("single")), mock.patch.object(
        gpa, "paged_attention_step_sharded", record("sharded")
    ), mock.patch.object(gpa, "kv_head_axes", return_value=("tensor",)):
      paged.forward_serve_gpu_paged(qkv, kv, kv, kv_pools=pools, metadata=page_table)

    self.assertIn("single", captured, "the unsharded mesh must reach paged_attention_step")
    self.assertIn("sharded", captured, "a KV-head-sharded mesh must reach paged_attention_step_sharded")
    self.assertEqual(
        captured["single"],
        captured["sharded"],
        "the sharded and single-device paged branches configure the kernel differently; "
        "they are the same attention and must pass identical scale, causal and backend",
    )
    # Pin the scale itself, so the two branches cannot agree on a wrong value.
    # MaxText has already folded depth scaling into the query projection.
    self.assertEqual(captured["sharded"]["scale"], 1.0)
    self.assertTrue(captured["sharded"]["causal"])

  def test_gpu_paged_dry_run_passes_through(self):
    """Model init and JIT tracing reach the branch with nothing to attend over."""
    seq_len = 128
    cfg = self._gpu_paged_config(seq_len, attention="gpu_paged")
    paged = self._build_attention(cfg, seq_len, MODEL_MODE_PREFILL)

    query = jnp.ones((self.global_batch_size, seq_len, cfg.num_query_heads, cfg.head_dim), cfg.dtype)
    out, cache = paged.forward_serve_gpu_paged(query, query, query, kv_pools=None, metadata=None)

    self.assertEqual(cache, [])
    self.assertEqual(out.shape, (self.global_batch_size * seq_len, cfg.num_query_heads, cfg.head_dim))


if __name__ == "__main__":
  unittest.main()
