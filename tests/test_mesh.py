# SPDX-License-Identifier: Apache-2.0
"""The serving mesh, whose axis names are dictated by MaxText's config."""

from __future__ import annotations

import pytest

from jax_vllm_plugin import mesh as mesh_module

jax = pytest.importorskip("jax", reason="the mesh is a JAX object")


class _Config:
    """The slice of MaxText's config that head-axis resolution reads.

    `layout_builder` resolves logical head axes through MaxText's own
    `logical_axis_rules` rather than inferring them from mesh axis names,
    because the serving recipe calls the tensor axis `model` while training
    recipes call it `tensor`. Only the rules are consulted here.
    """

    logical_axis_rules = (
        ("activation_kv_heads", mesh_module.TENSOR_AXIS),
        ("activation_heads", mesh_module.TENSOR_AXIS),
    )


CONFIG = _Config()


def test_layout_builder_compatibility_accepts_mesh_only_api():
    calls = []

    result = mesh_module._call_maxtext_layout(
        lambda mesh: calls.append(mesh) or 2,
        CONFIG,
        "mesh",
    )

    assert result == 2
    assert calls == ["mesh"]


def test_layout_builder_compatibility_passes_config_to_new_api():
    calls = []

    def config_aware(config, mesh):
        calls.append((config, mesh))
        return 4

    result = mesh_module._call_maxtext_layout(
        config_aware,
        CONFIG,
        "mesh",
    )

    assert result == 4
    assert calls == [(CONFIG, "mesh")]


def test_the_mesh_carries_every_axis_maxtexts_rules_name():
    """MaxText applies `inference/vllm.yml`'s logical_axis_rules against this
    mesh. An axis those rules name but the mesh lacks does not raise -- it
    resolves to no sharding, which silently replicates what should be split."""
    mesh = mesh_module.build_mesh(1, devices=jax.devices()[:1])

    assert tuple(mesh.shape) == mesh_module.MESH_AXES


def test_tensor_parallelism_lands_on_the_axis_maxtext_shards_heads_over():
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
    mesh = mesh_module.build_mesh(2, devices=devices[:2])

    assert dict(mesh.shape)[mesh_module.TENSOR_AXIS] == 2


def test_asking_for_more_devices_than_exist_is_refused():
    with pytest.raises(ValueError, match="needs that many devices"):
        mesh_module.build_mesh(len(jax.devices()) + 1)


def test_a_non_positive_tensor_parallel_size_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        mesh_module.build_mesh(0)


def test_a_wider_host_is_not_silently_used_in_full():
    """A request for two devices on an eight-device host should shard two ways,
    not eight. Widening the mesh would change the sharding under the caller."""
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
    mesh = mesh_module.build_mesh(2, devices=devices)

    assert mesh.devices.size == 2


def test_kv_head_shards_agrees_with_maxtext_about_this_meshs_naming():
    """The vLLM mesh names its tensor axis `model`, and MaxText's paged path
    historically looked only for `tensor`. If these disagree the pool is sharded
    on axes the kernels do not know about, which no shape check would catch."""
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("needs at least two devices")
    mesh = mesh_module.build_mesh(2, devices=devices[:2])

    assert mesh_module.kv_head_shards(CONFIG, mesh) == 2


def test_a_head_count_that_does_not_divide_the_mesh_is_refused():
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("needs at least four devices")
    mesh = mesh_module.build_mesh(4, devices=devices[:4])

    with pytest.raises(ValueError, match="does not divide evenly"):
        mesh_module.assert_kv_heads_shardable(CONFIG, mesh, num_kv_heads=6)


def test_fewer_heads_than_shards_is_allowed_but_says_so(caplog):
    """MaxText replicates rather than failing, which is correct and multiplies
    the pool's footprint -- so it should be a decision, not an OOM."""
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("needs at least four devices")
    mesh = mesh_module.build_mesh(4, devices=devices[:4])

    with caplog.at_level("WARNING", logger="jax_vllm_plugin.mesh"):
        mesh_module.assert_kv_heads_shardable(CONFIG, mesh, num_kv_heads=2)

    assert "replicated 2 times" in caplog.text
