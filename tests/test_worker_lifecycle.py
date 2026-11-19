# SPDX-License-Identifier: Apache-2.0
"""Worker initialization resource-lifecycle contracts."""

from __future__ import annotations

import contextlib
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm", reason="vLLM is supplied by the runtime image")

from jax_vllm_gpu.plugin.worker import JaxWorker


def test_distributed_init_restores_config_and_removes_rendezvous_on_error(monkeypatch):
    import vllm.config
    import vllm.distributed.parallel_state

    rendezvous_path = None

    def fail_init(*, distributed_init_method, **_kwargs):
        nonlocal rendezvous_path
        rendezvous_path = distributed_init_method.removeprefix("file://")
        assert os.path.exists(rendezvous_path)
        raise RuntimeError("injected initialization failure")

    monkeypatch.setattr(
        vllm.config, "set_current_vllm_config", lambda _config: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        vllm.distributed.parallel_state, "init_distributed_environment", fail_init
    )

    worker = JaxWorker.__new__(JaxWorker)
    worker.vllm_config = object()
    worker.parallel_config = SimpleNamespace(data_parallel_size=4)

    with pytest.raises(RuntimeError, match="injected"):
        worker._init_torch_distributed()

    assert worker.parallel_config.data_parallel_size == 4
    assert rendezvous_path is not None
    assert not os.path.exists(rendezvous_path)
