from __future__ import annotations

from collections.abc import Iterable

import jraph
import numpy as np

from ..atomic import AtomicNumberTable
from ..format_hdf5 import HDF5Dataset
from ..statistics_data import (
    compute_average_atomic_energies as _compute_average_atomic_energies,
)


def compute_atomic_numbers(dataset: HDF5Dataset) -> AtomicNumberTable:
    """Collect the set of chemical species present in an HDF5 dataset."""

    z_set: set[int] = set()
    for idx in range(len(dataset)):
        atoms = dataset[idx]
        z_set.update(int(z) for z in atoms.get_atomic_numbers())

    return AtomicNumberTable(sorted(z_set))


def compute_average_atomic_energies(
    dataset: HDF5Dataset,
    z_table: AtomicNumberTable,
    max_n: int | None = None,
) -> dict[int, float]:
    return _compute_average_atomic_energies(dataset, z_table, max_n=max_n)


def compute_statistics(
    graphs: Iterable[jraph.GraphsTuple],
    atomic_energies: dict[int, float],
    atomic_numbers: AtomicNumberTable,
) -> tuple[float, float, float]:
    """Compute dataset statistics for JAX graph batches.

    Parameters
    ----------
    graphs
        Iterable returning ``jraph.GraphsTuple`` batches, typically a
        :class:`mace_jax.data.utils.GraphDataLoader`.
    atomic_energies
        Mapping from atomic number to reference energy.
    atomic_numbers
        Atomic number table describing the ordering used for species indices.
    """

    atomic_numbers = list(atomic_numbers or [])
    if not atomic_numbers:
        raise ValueError('Atomic numbers table must contain at least one entry.')

    energy_lookup = np.asarray([atomic_energies[int(z)] for z in atomic_numbers])

    per_atom_residuals: list[float] = []
    force_rows: list[np.ndarray] = []
    neighbor_counts: list[np.ndarray] = []
    processed_any = False

    graph_iterable: Iterable[jraph.GraphsTuple]
    if hasattr(graphs, 'graphs'):
        graph_iterable = getattr(graphs, 'graphs')
    else:
        graph_iterable = graphs

    for item in graph_iterable:
        if item is None:
            continue

        processed_any = True
        if isinstance(item, jraph.GraphsTuple) and np.asarray(item.n_node).size > 1:
            sub_graphs = jraph.unbatch(item)
        else:
            sub_graphs = [item]

        for graph in sub_graphs:
            species_idx = np.asarray(graph.nodes.species)
            if species_idx.size == 0:
                continue

            if species_idx.min(initial=0) < 0 or species_idx.max(initial=0) >= len(
                energy_lookup
            ):
                raise ValueError('Species indices exceed atomic_numbers table.')

            node_e0 = energy_lookup[species_idx]

            graph_energy = float(np.asarray(graph.globals.energy).reshape(-1)[0])
            graph_size = species_idx.shape[0]

            per_atom_residuals.append(
                (graph_energy - float(np.sum(node_e0))) / graph_size
            )

            forces = np.asarray(
                getattr(
                    graph.nodes,
                    'forces',
                    np.zeros((graph_size, 3), dtype=np.float64),
                )
            )
            if forces.size:
                force_rows.append(forces.reshape(-1, 3))

            receivers = np.asarray(graph.receivers)
            if receivers.size:
                counts = np.bincount(receivers, minlength=graph_size)
            else:
                counts = np.zeros(graph_size, dtype=np.int32)
            neighbor_counts.append(counts.astype(np.float64))

    if not processed_any or not per_atom_residuals:
        raise ValueError('No graphs available to compute statistics.')

    atom_residuals = np.asarray(per_atom_residuals, dtype=np.float64)
    mean = float(atom_residuals.mean())

    if force_rows:
        forces_all = np.concatenate(force_rows, axis=0).astype(np.float64, copy=False)
        rms = float(np.sqrt(np.mean(np.square(forces_all))))
    else:
        rms = 0.0

    if neighbor_counts:
        neighbors_all = np.concatenate(neighbor_counts, axis=0)
        avg_neighbors = float(neighbors_all.mean())
    else:
        avg_neighbors = 0.0

    return avg_neighbors, mean, rms


__all__ = [
    'compute_atomic_numbers',
    'compute_average_atomic_energies',
    'compute_statistics',
]
