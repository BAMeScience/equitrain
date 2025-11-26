from __future__ import annotations

from pathlib import Path

import numpy as np

from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_jax.loaders import get_dataloader
from equitrain.data.format_hdf5 import HDF5Dataset


def _dataset_metadata(path: Path) -> tuple[AtomicNumberTable, int]:
    z_set: set[int] = set()
    max_nodes = 1
    with HDF5Dataset(path, mode='r') as dataset:
        for idx in range(len(dataset)):
            atoms = dataset[idx]
            z_set.update(int(z) for z in atoms.get_atomic_numbers())
            max_nodes = max(max_nodes, len(atoms))
    return AtomicNumberTable(sorted(z_set)), max_nodes


def _collect_sequence(loader) -> list[float]:
    sequence: list[float] = []
    for batch in loader:
        graphs = batch if isinstance(batch, list) else (batch,)
        for graph in graphs:
            energy = np.asarray(graph.globals['energy']).reshape(-1)[0]
            sequence.append(float(energy))
    return sequence


def test_graph_data_loader_deterministic_shuffle():
    data_file = Path(__file__).with_name('data') / 'train.h5'
    z_table, max_nodes = _dataset_metadata(data_file)
    max_edges = max_nodes * max_nodes

    loader_a = get_dataloader(
        data_file=data_file,
        atomic_numbers=z_table,
        r_max=3.0,
        batch_size=1,
        shuffle=True,
        max_nodes=max_nodes,
        max_edges=max_edges,
        seed=2025,
    )
    loader_b = get_dataloader(
        data_file=data_file,
        atomic_numbers=z_table,
        r_max=3.0,
        batch_size=1,
        shuffle=True,
        max_nodes=max_nodes,
        max_edges=max_edges,
        seed=2025,
    )
    loader_c = get_dataloader(
        data_file=data_file,
        atomic_numbers=z_table,
        r_max=3.0,
        batch_size=1,
        shuffle=True,
        max_nodes=max_nodes,
        max_edges=max_edges,
        seed=1337,
    )

    seq_a = _collect_sequence(loader_a)
    seq_b = _collect_sequence(loader_b)
    seq_c = _collect_sequence(loader_c)

    assert seq_a == seq_b
    assert seq_a != seq_c
