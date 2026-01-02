from __future__ import annotations

import jax
import jax.numpy as jnp
import jraph
import numpy as np
from jax import tree_util as jtu

from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.configuration import Configuration
from equitrain.data.neighborhood import get_neighborhood


class _AttrDict(dict):
    """Dictionary with attribute-style access used inside GraphsTuple."""

    __slots__ = ()

    def __getattr__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc


def _attrdict_flatten(obj: _AttrDict):
    keys = tuple(obj.keys())
    return [obj[key] for key in keys], keys


def _attrdict_unflatten(keys, values):
    return _AttrDict({key: value for key, value in zip(keys, values)})


jtu.register_pytree_node(_AttrDict, _attrdict_flatten, _attrdict_unflatten)


def graph_from_configuration(
    config: Configuration,
    *,
    cutoff: float,
    z_table: AtomicNumberTable,
) -> jraph.GraphsTuple:
    positions = np.asarray(config.positions, dtype=np.float32)
    num_atoms = int(positions.shape[0])
    if num_atoms <= 0:
        raise ValueError('Configurations without atoms cannot be converted to graphs.')

    atomic_numbers = np.asarray(config.atomic_numbers, dtype=np.int32)
    species = _species_indices(z_table, atomic_numbers)

    forces = np.asarray(config.forces, dtype=np.float32)
    if forces.shape != positions.shape:
        forces = np.zeros_like(positions, dtype=np.float32)

    charges = None
    if config.charges is not None:
        charges = np.asarray(config.charges, dtype=np.float32).reshape(num_atoms, -1)

    cell = _safe_cell(config.cell)
    pbc = tuple(bool(x) for x in (config.pbc or (False, False, False)))
    senders, receivers, shifts, unit_shifts = _build_edges(
        positions, cutoff, pbc=pbc, cell=cell
    )

    globals_dict = _AttrDict(
        cell=cell[None, ...],
        energy=_scalar_array(config.energy),
        weight=_scalar_array(config.energy_weight),
        energy_weight=_scalar_array(config.energy_weight),
        forces_weight=_scalar_array(config.forces_weight),
        stress_weight=_scalar_array(config.stress_weight),
        virials_weight=_scalar_array(config.virials_weight),
        dipole_weight=_scalar_array(config.dipole_weight),
        stress=_matrix_array(_voigt_to_full(config.stress)),
        virials=_matrix_array(config.virials),
        dipole=_vector_array(config.dipole),
    )

    nodes = _AttrDict(
        positions=positions,
        forces=forces,
        species=species,
    )
    if charges is not None:
        nodes['charges'] = charges

    edges = _AttrDict(
        shifts=shifts,
        unit_shifts=unit_shifts,
    )

    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=globals_dict,
        n_node=np.asarray([num_atoms], dtype=np.int32),
        n_edge=np.asarray([receivers.shape[0]], dtype=np.int32),
    )


def _build_edges(
    positions: np.ndarray,
    cutoff: float,
    *,
    pbc: tuple[bool, bool, bool],
    cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if positions.size == 0 or cutoff <= 0.0:
        empty = np.zeros((0,), dtype=np.int32)
        zero_vec = np.zeros((0, 3), dtype=np.float32)
        return empty, empty, zero_vec, zero_vec

    edge_index, shifts, unit_shifts, _ = get_neighborhood(
        positions.astype(np.float64, copy=False),
        cutoff=cutoff,
        pbc=pbc,
        cell=cell.astype(np.float64, copy=True),
    )

    if edge_index.size == 0:
        empty = np.zeros((0,), dtype=np.int32)
        zero_vec = np.zeros((0, 3), dtype=np.float32)
        return empty, empty, zero_vec, zero_vec

    senders, receivers = edge_index.astype(np.int32, copy=False)
    shifts = np.asarray(shifts, dtype=np.float32)
    unit_shifts = np.asarray(unit_shifts, dtype=np.float32)
    return senders, receivers, shifts, unit_shifts


def _species_indices(
    z_table: AtomicNumberTable, atomic_numbers: np.ndarray
) -> np.ndarray:
    indices = np.empty_like(atomic_numbers, dtype=np.int32)
    for idx, number in enumerate(atomic_numbers):
        try:
            indices[idx] = z_table.z_to_index(int(number))
        except ValueError as exc:  # pragma: no cover - safeguarded upstream
            raise ValueError(
                f'Atomic number {int(number)} not present in the atomic numbers table.'
            ) from exc
    return indices


def _safe_cell(cell: np.ndarray | None) -> np.ndarray:
    if cell is None:
        return np.eye(3, dtype=np.float32)
    arr = np.asarray(cell, dtype=np.float32)
    if arr.shape != (3, 3):
        raise ValueError('Lattice cell must have shape (3, 3).')
    return arr


def _scalar_array(value: float | int | None) -> np.ndarray:
    if value is None:
        value = 0.0
    return np.asarray([float(value)], dtype=np.float32)


def _matrix_array(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        base = np.zeros((3, 3), dtype=np.float32)
    else:
        base = np.asarray(value, dtype=np.float32).reshape(3, 3)
    return base[None, ...]


def _vector_array(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        base = np.zeros(3, dtype=np.float32)
    else:
        base = np.asarray(value, dtype=np.float32).reshape(3)
    return base[None, ...]


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
    graph_mask = jraph.get_graph_padding_mask(graph)
    all_positive = jnp.all(n_node > 0)
    graph_mask = jnp.where(
        all_positive,
        jnp.ones_like(graph_mask, dtype=graph_mask.dtype),
        graph_mask,
    )
    node_mask = jnp.repeat(
        graph_mask, graph.n_node, total_repeat_length=positions.shape[0]
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
    ptr = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(graph.n_node.astype(jnp.int32)),
        ]
    )

    data_dict: dict[str, jnp.ndarray] = {
        'positions': positions,
        'node_attrs': node_attrs,
        'node_attrs_index': species,
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

    for field in (
        'weight',
        'energy_weight',
        'forces_weight',
        'stress_weight',
        'virials_weight',
        'dipole_weight',
    ):
        value = getattr(graph.globals, field, None)
        if value is not None:
            data_dict[field] = jnp.asarray(value, dtype=positions.dtype)

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
