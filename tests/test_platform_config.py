# SPDX-License-Identifier: Apache-2.0
"""What the platform forces into vLLM's config at startup."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm", reason="vLLM is supplied by the runtime image")

from jax_vllm_plugin.platform import (
    ASYNC_SCHEDULING_ENV_VAR,
    DEFAULT_TOKENS_PER_PAGE,
    JaxPlatform,
)


def make_config(**kwargs):
    parallel = SimpleNamespace(
        worker_cls="",
        pipeline_parallel_size=kwargs.pop("pipeline_parallel_size", 1),
        distributed_executor_backend=None,
    )
    cache = SimpleNamespace(
        user_specified_block_size=kwargs.pop("user_specified_block_size", False),
        block_size=kwargs.pop("block_size", 0),
        cache_dtype=kwargs.pop("cache_dtype", "auto"),
    )
    scheduler = SimpleNamespace(
        async_scheduling=kwargs.pop("async_scheduling", True),
    )
    model = SimpleNamespace(
        max_model_len=kwargs.pop("max_model_len", 2048),
        quantization=kwargs.pop("quantization", None),
    )
    load_format = kwargs.pop("load_format", None)
    return SimpleNamespace(
        parallel_config=parallel,
        cache_config=cache,
        scheduler_config=scheduler,
        model_config=model,
        additional_config=kwargs.pop("additional_config", {}),
        speculative_config=kwargs.pop("speculative_config", None),
        lora_config=kwargs.pop("lora_config", None),
        load_config=(
            None if load_format is None else SimpleNamespace(load_format=load_format)
        ),
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


def test_async_scheduling_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(ASYNC_SCHEDULING_ENV_VAR, raising=False)
    config = make_config(async_scheduling=True)
    JaxPlatform.check_and_update_config(config)

    assert config.scheduler_config.async_scheduling is False


def test_async_scheduling_can_be_enabled_explicitly(monkeypatch):
    monkeypatch.setenv(ASYNC_SCHEDULING_ENV_VAR, "1")
    config = make_config(async_scheduling=True)
    JaxPlatform.check_and_update_config(config)

    assert config.scheduler_config.async_scheduling is True


def test_pipeline_parallelism_is_rejected():
    config = make_config(pipeline_parallel_size=2)

    with pytest.raises(ValueError, match="pipeline parallelism is unsupported"):
        JaxPlatform.check_and_update_config(config)


def test_block_size_defaults_to_the_pool_page_size():
    config = make_config()
    JaxPlatform.check_and_update_config(config)

    assert config.cache_config.block_size == DEFAULT_TOKENS_PER_PAGE


def test_an_explicit_block_size_is_respected():
    config = make_config(user_specified_block_size=True, block_size=32)
    JaxPlatform.check_and_update_config(config)

    assert config.cache_config.block_size == 32


def test_an_unsupported_block_size_is_rejected():
    config = make_config(user_specified_block_size=True, block_size=64)

    with pytest.raises(ValueError, match="block sizes 16 and 32"):
        JaxPlatform.check_and_update_config(config)


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


def test_maxtexts_context_bound_covers_what_vllm_will_admit():
    """MaxText validates a request against its own max_target_length. Left at the
    default that is shorter than vLLM's limit, so the model would reject a long
    request the scheduler had already accepted."""
    config = make_config(max_model_len=32768)
    JaxPlatform.check_and_update_config(config)

    assert config.additional_config["maxtext_config"]["max_target_length"] == 32768


def test_maxtext_gets_a_block_count_large_enough_for_one_full_request():
    """The number is inert -- vLLM owns allocation and MaxText skips
    init_kv_caches in this mode -- but gpu_paged validation still rejects a
    figure too small to hold a single longest request."""
    config = make_config(max_model_len=2048, user_specified_block_size=True, block_size=32)
    JaxPlatform.check_and_update_config(config)

    assert config.additional_config["maxtext_config"]["paged_num_blocks"] == 64


def test_an_incompatible_attention_choice_is_rejected():
    config = make_config(additional_config={"maxtext_config": {"attention": "dot_product"}})

    with pytest.raises(ValueError, match="requires MaxText attention='gpu_paged'"):
        JaxPlatform.check_and_update_config(config)


def test_mismatched_page_sizes_are_rejected():
    config = make_config(
        block_size=16,
        user_specified_block_size=True,
        additional_config={"maxtext_config": {"paged_page_size": 32}},
    )

    with pytest.raises(ValueError, match="must match vLLM block_size"):
        JaxPlatform.check_and_update_config(config)


def test_a_config_without_additional_config_gets_the_required_settings():
    config = make_config()
    config.additional_config = None
    JaxPlatform.check_and_update_config(config)

    assert config.additional_config["maxtext_config"]["attention"] == "gpu_paged"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("speculative_config", object(), "speculative decoding"),
        ("lora_config", object(), "LoRA"),
        ("quantization", "fp8", "quantization"),
        ("cache_dtype", "fp8", "KV cache dtype"),
    ],
)
def test_unsupported_engine_features_fail_at_startup(field, value, message):
    config = make_config(**{field: value})

    with pytest.raises(ValueError, match=message):
        JaxPlatform.check_and_update_config(config)


def test_a_checkpoint_is_required_for_a_non_dummy_load():
    config = make_config(load_format="auto")

    with pytest.raises(ValueError, match="refusing to serve random weights"):
        JaxPlatform.check_and_update_config(config)


def test_dummy_load_explicitly_allows_random_weights():
    config = make_config(load_format="dummy")

    JaxPlatform.check_and_update_config(config)


def test_a_maxtext_checkpoint_satisfies_the_load_contract():
    config = make_config(
        load_format="auto",
        additional_config={
            "maxtext_config": {"load_parameters_path": "/model/checkpoint"}
        },
    )

    JaxPlatform.check_and_update_config(config)
