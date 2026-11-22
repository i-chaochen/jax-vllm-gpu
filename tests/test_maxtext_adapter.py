# SPDX-License-Identifier: Apache-2.0
"""The MaxText adapter binds this repository's paged attention into MaxText.

If the binding were missing, MaxText would run its own copy of the paged
attention step, which builds a different `PagedPlan` class: plans built here
would fail its isinstance check and be rebuilt from padded block tables, writing
junk KV into real pages. So the binding is asserted, not assumed.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("maxtext", reason="MaxText is supplied by a pinned checkout")

from maxtext.layers import attentions

from vllm_xla_gpu.models.adapters import maxtext


def test_install_binds_this_repositorys_implementation():
    maxtext.install()

    assert attentions.Attention.forward_serve_gpu_paged is maxtext.forward_serve_gpu_paged
    assert maxtext.installed()


def test_install_is_idempotent():
    maxtext.install()
    maxtext.install()

    assert maxtext.installed()


def test_install_refuses_a_maxtext_without_the_hook(monkeypatch):
    monkeypatch.delattr(attentions.Attention, maxtext.HOOK)

    with pytest.raises(RuntimeError, match="gpu_paged"):
        maxtext.install()


def test_registering_maxtext_serves_this_adapter_without_tpu_inference():
    pytest.importorskip("vllm", reason="vLLM is supplied by the runtime image")
    from vllm_xla_gpu.plugin import registry

    registry.register_maxtext()

    model_cls = registry.get_model_cls("MaxTextForCausalLM")
    assert model_cls.__module__ == "vllm_xla_gpu.models.adapters.maxtext_vllm"
    assert maxtext.installed()
    assert "tpu_inference" not in sys.modules
