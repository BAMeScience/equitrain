from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import jraph
import numpy as np
from mace_jax.data.utils import AtomicNumberTable as JaxAtomicNumberTable
from mace_jax.data.utils import Configuration as JaxConfiguration
from mace_jax.data.utils import graph_from_configuration

from equitrain.data.configuration import Configuration as EqConfiguration
from equitrain.data.format_hdf5.dataset import HDF5Dataset


def atoms_to_graphs(
    data_path: Path | str,
    r_max: float,
    z_table: JaxAtomicNumberTable,
) -> list[jraph.GraphsTuple]:
    if data_path is None:
        return []

    dataset = HDF5Dataset(data_path, mode='r')
    graphs: list[jraph.GraphsTuple] = []
    try:
        for idx in range(len(dataset)):
            atoms = dataset[idx]
            eq_conf = EqConfiguration.from_atoms(atoms)
            jax_conf = JaxConfiguration(
                atomic_numbers=eq_conf.atomic_numbers,
                positions=np.asarray(eq_conf.positions, dtype=np.float32),
                energy=np.array(eq_conf.energy, dtype=np.float32),
                forces=np.asarray(eq_conf.forces, dtype=np.float32),
                stress=_voigt_to_full(eq_conf.stress),
                cell=np.asarray(eq_conf.cell, dtype=np.float32),
                pbc=tuple(bool(x) for x in eq_conf.pbc),
                weight=np.array(eq_conf.energy_weight, dtype=np.float32),
            )
            graph = graph_from_configuration(
                jax_conf,
                cutoff=float(r_max),
                z_table=z_table,
            )
            graphs.append(graph)
    finally:
        dataset.close()
    return graphs


def _voigt_to_full(stress: np.ndarray | None) -> np.ndarray | None:
    if stress is None:
        return None
    stress = np.asarray(stress)
    if stress.shape == (3, 3):
        return stress
    if stress.shape != (6,):
        return None

    sxx, syy, szz, syz, sxz, sxy = stress
    return np.array(
        [
            [sxx, sxy, sxz],
            [sxy, syy, syz],
            [sxz, syz, szz],
        ],
        dtype=stress.dtype,
    )


def graph_to_data(graph: jraph.GraphsTuple, num_species: int) -> dict[str, jnp.ndarray]:
    positions = jnp.asarray(graph.nodes.positions, dtype=jnp.float32)
    shifts = jnp.asarray(graph.edges.shifts, dtype=positions.dtype)
    cell = jnp.asarray(graph.globals.cell, dtype=positions.dtype)

    species = jnp.asarray(graph.nodes.species, dtype=jnp.int32)
    senders = jnp.asarray(graph.senders, dtype=jnp.int32)
    receivers = jnp.asarray(graph.receivers, dtype=jnp.int32)

    n_node = jnp.asarray(graph.n_node, dtype=jnp.int32)
    mask = jraph.get_graph_padding_mask(graph).astype(jnp.float32)
    node_mask = jnp.repeat(
        mask, graph.n_node, total_repeat_length=positions.shape[0]
    ).astype(positions.dtype)

    node_attrs = jax.nn.one_hot(
        species,
        num_classes=num_species,
        dtype=positions.dtype,
    )
    node_attrs = node_attrs * node_mask[:, None]

    graph_indices = jnp.arange(graph.n_node.shape[0], dtype=jnp.int32)
    batch = jnp.repeat(
        graph_indices, graph.n_node, total_repeat_length=positions.shape[0]
    )
    ptr = jnp.concatenate([
        jnp.array([0], dtype=jnp.int32),
        jnp.cumsum(graph.n_node.astype(jnp.int32)),
    ])

    data_dict: dict[str, jnp.ndarray] = {
        'positions': positions,
        'node_attrs': node_attrs,
        'edge_index': jnp.stack([senders, receivers], axis=0),
        'shifts': shifts,
        'batch': batch,
        'ptr': ptr,
        'cell': cell,
    }

    unit_shifts = getattr(graph.edges, 'unit_shifts', None)
    if unit_shifts is None:
        unit_shifts = jnp.zeros(shifts.shape, dtype=positions.dtype)
    else:
        unit_shifts = jnp.asarray(unit_shifts, dtype=positions.dtype)
    data_dict['unit_shifts'] = unit_shifts

    if hasattr(graph.nodes, 'head'):
        data_dict['head'] = graph.nodes.head

    return data_dict


def make_apply_fn(wrapper, num_species: int):
    def apply_fn(variables, graph):
        data_dict = graph_to_data(graph, num_species=num_species)
        return wrapper.apply(
            variables,
            data_dict,
            compute_force=wrapper.compute_force,
            compute_stress=wrapper.compute_stress,
        )

    return apply_fn
