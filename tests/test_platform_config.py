# SPDX-License-Identifier: Apache-2.0
"""What the platform forces into vLLM's config at startup."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm", reason="vLLM is supplied by the runtime image")

from jax_vllm_plugin.platform import DEFAULT_TOKENS_PER_PAGE, JaxPlatform


def make_config(**kwargs):
    parallel = SimpleNamespace(
        worker_cls="",
        pipeline_parallel_size=kwargs.pop("pipeline_parallel_size", 1),
        distributed_executor_backend=None,
    )
    cache = SimpleNamespace(
        user_specified_block_size=kwargs.pop("user_specified_block_size", False),
        block_size=kwargs.pop("block_size", 0),
    )
    return SimpleNamespace(
        parallel_config=parallel,
        cache_config=cache,
        additional_config=kwargs.pop("additional_config", {}),
    )


def test_the_worker_class_is_named():
    config = make_config()
    JaxPlatform.check_and_update_config(config)

    assert config.parallel_config.worker_cls == "jax_vllm_plugin.worker.JaxWorker"


def test_a_single_pipeline_stage_uses_the_uniproc_executor():
    """One worker drives every GPU through a JAX mesh, so vLLM must not fan out
    into a process per device."""
    config = make_config()
    JaxPlatform.check_and_update_config(config)

    assert config.parallel_config.distributed_executor_backend == "uni"


def test_pipeline_parallelism_is_left_alone():
    config = make_config(pipeline_parallel_size=2)
    JaxPlatform.check_and_update_config(config)

    assert config.parallel_config.distributed_executor_backend is None


def test_block_size_defaults_to_the_pool_page_size():
    config = make_config()
    JaxPlatform.check_and_update_config(config)

    assert config.cache_config.block_size == DEFAULT_TOKENS_PER_PAGE


def test_an_explicit_block_size_is_respected():
    config = make_config(user_specified_block_size=True, block_size=32)
    JaxPlatform.check_and_update_config(config)

    assert config.cache_config.block_size == 32


def test_maxtext_is_pointed_at_gpu_paged_attention():
    """MaxText's inference/vllm.yml defaults to vllm_rpa, which is TPU Pallas.
    Without this the model would route around the AMD paged kernels entirely."""
    config = make_config()
    JaxPlatform.check_and_update_config(config)

    assert config.additional_config["maxtext_config"]["attention"] == "gpu_paged"


def test_the_page_size_handed_to_maxtext_matches_vllms_block_size():
    """Two names for one number. If they disagree, the page table indexes pages
    of a size the pool was not built with."""
    config = make_config(user_specified_block_size=True, block_size=32)
    JaxPlatform.check_and_update_config(config)

    assert config.additional_config["maxtext_config"]["paged_page_size"] == 32


def test_an_explicit_attention_choice_is_not_overridden():
    config = make_config(additional_config={"maxtext_config": {"attention": "dot_product"}})
    JaxPlatform.check_and_update_config(config)

    assert config.additional_config["maxtext_config"]["attention"] == "dot_product"


def test_a_config_without_additional_config_does_not_crash():
    config = make_config()
    config.additional_config = None
    JaxPlatform.check_and_update_config(config)

    assert config.parallel_config.worker_cls
