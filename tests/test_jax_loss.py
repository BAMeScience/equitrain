from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('jax')
import jax.numpy as jnp
import jraph

from equitrain.backends.jax_loss import LossSettings, build_loss_fn


def _make_graph(
    *,
    num_nodes: int = 2,
    energy: float = 0.0,
    weight: float = 1.0,
    forces: jnp.ndarray | None = None,
    stress: jnp.ndarray | None = None,
) -> jraph.GraphsTuple:
    pad_nodes = num_nodes
    total_nodes = num_nodes + pad_nodes

    positions = jnp.zeros((total_nodes, 3), dtype=jnp.float32)
    if forces is None:
        forces = jnp.zeros((num_nodes, 3), dtype=jnp.float32)
    forces = jnp.concatenate(
        [jnp.asarray(forces, dtype=jnp.float32), jnp.zeros((pad_nodes, 3), dtype=jnp.float32)],
        axis=0,
    )
    species = jnp.concatenate(
        [jnp.zeros((num_nodes,), dtype=jnp.int32), jnp.zeros((pad_nodes,), dtype=jnp.int32)],
        axis=0,
    )

    stress_tensor = jnp.zeros((2, 3, 3), dtype=jnp.float32)
    if stress is not None:
        stress_tensor = stress_tensor.at[0].set(jnp.asarray(stress, dtype=jnp.float32))

    nodes = SimpleNamespace(
        positions=positions,
        forces=forces,
        species=species,
    )
    edges = SimpleNamespace(
        shifts=jnp.zeros((0, 3), dtype=jnp.float32),
        unit_shifts=jnp.zeros((0, 3), dtype=jnp.float32),
    )
    globals_ = SimpleNamespace(
        cell=jnp.eye(3, dtype=jnp.float32),
        energy=jnp.asarray([energy, 0.0], dtype=jnp.float32),
        stress=stress_tensor,
        weight=jnp.asarray([weight, 0.0], dtype=jnp.float32),
    )

    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        globals=globals_,
        senders=jnp.zeros((0,), dtype=jnp.int32),
        receivers=jnp.zeros((0,), dtype=jnp.int32),
        n_node=jnp.asarray([num_nodes, pad_nodes], dtype=jnp.int32),
        n_edge=jnp.asarray([0, 0], dtype=jnp.int32),
    )


def _loss_settings(**overrides) -> LossSettings:
    base = LossSettings()
    return dataclasses.replace(
        base,
        loss_type='mse',
        loss_type_energy='mse',
        loss_type_forces='mse',
        loss_type_stress='mse',
        smooth_l1_beta=1.0,
        huber_delta=1.0,
        **overrides,
    )


def test_build_loss_fn_energy_component():
    graph = _make_graph(energy=1.0)

    def apply_fn(_, _graph):
        total_nodes = graph.nodes.positions.shape[0]
        return {
            'energy': jnp.asarray([3.0, 0.0], dtype=jnp.float32),
            'forces': jnp.zeros((total_nodes, 3), dtype=jnp.float32),
            'stress': jnp.zeros((2, 3, 3), dtype=jnp.float32),
        }

    settings = _loss_settings(
        forces_weight=0.0,
        stress_weight=0.0,
        loss_energy_per_atom=False,
    )
    loss_fn = build_loss_fn(apply_fn, settings)
    loss_value, aux = loss_fn(None, graph)

    assert pytest.approx(float(loss_value)) == 4.0

    energy_loss, energy_count = aux['metrics']['energy']
    total_loss, total_count = aux['metrics']['total']

    np.testing.assert_allclose(np.asarray(energy_loss), 4.0)
    np.testing.assert_allclose(np.asarray(energy_count), 1.0)
    np.testing.assert_allclose(np.asarray(total_loss), 4.0)
    np.testing.assert_allclose(np.asarray(total_count), 1.0)


def test_build_loss_fn_forces_component():
    graph = _make_graph(forces=jnp.zeros((2, 3)))

    def apply_fn(_, _graph):
        total_nodes = graph.nodes.positions.shape[0]
        n_real = int(graph.n_node[0])
        real_forces = jnp.ones((n_real, 3), dtype=jnp.float32)
        padded_forces = jnp.zeros((total_nodes - n_real, 3), dtype=jnp.float32)
        forces = jnp.concatenate([real_forces, padded_forces], axis=0)
        return {
            'energy': jnp.asarray([0.0, 0.0], dtype=jnp.float32),
            'forces': forces,
            'stress': jnp.zeros((2, 3, 3), dtype=jnp.float32),
        }

    settings = _loss_settings(
        energy_weight=0.0,
        forces_weight=0.5,
        stress_weight=0.0,
        loss_energy_per_atom=False,
    )
    loss_fn = build_loss_fn(apply_fn, settings)
    loss_value, aux = loss_fn(None, graph)

    assert pytest.approx(float(loss_value)) == 0.5

    forces_loss, forces_count = aux['metrics']['forces']
    total_loss, total_count = aux['metrics']['total']

    np.testing.assert_allclose(np.asarray(forces_loss), 1.0)
    np.testing.assert_allclose(np.asarray(forces_count), 6.0)
    np.testing.assert_allclose(np.asarray(total_loss), 0.5)
    np.testing.assert_allclose(np.asarray(total_count), 1.0)


def test_build_loss_fn_stress_component():
    graph = _make_graph(stress=jnp.zeros((3, 3)))

    def apply_fn(_, _graph):
        total_nodes = graph.nodes.positions.shape[0]
        return {
            'energy': jnp.asarray([0.0, 0.0], dtype=jnp.float32),
            'forces': jnp.zeros((total_nodes, 3), dtype=jnp.float32),
            'stress': jnp.stack([
                jnp.ones((3, 3), dtype=jnp.float32),
                jnp.zeros((3, 3), dtype=jnp.float32),
            ], axis=0),
        }

    settings = _loss_settings(
        energy_weight=0.0,
        forces_weight=0.0,
        stress_weight=2.0,
        loss_energy_per_atom=False,
    )
    loss_fn = build_loss_fn(apply_fn, settings)
    loss_value, aux = loss_fn(None, graph)

    assert pytest.approx(float(loss_value)) == 2.0

    stress_loss, stress_count = aux['metrics']['stress']
    total_loss, total_count = aux['metrics']['total']

    np.testing.assert_allclose(np.asarray(stress_loss), 1.0)
    np.testing.assert_allclose(np.asarray(stress_count), 9.0)
    np.testing.assert_allclose(np.asarray(total_loss), 2.0)
    np.testing.assert_allclose(np.asarray(total_count), 1.0)
