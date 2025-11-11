from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch_geometric

pytest.importorskip('jax', reason='JAX runtime is required for JAX backend tests.')

from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_jax import atoms_to_graphs, build_loader
from equitrain.data.backend_jax.statistics import (
    compute_statistics as compute_statistics_jax,
)
from equitrain.data.backend_torch.statistics import (
    compute_statistics as compute_statistics_torch,
)
from equitrain.data.format_hdf5 import HDF5GraphDataset
from equitrain.data.statistics_data import Statistics


def test_backend_jax_statistics_matches_torch():
    data_dir = Path(__file__).with_name('data')
    train_file = data_dir / 'train.h5'
    stats = Statistics.load(data_dir / 'statistics.json')

    with HDF5GraphDataset(
        train_file,
        r_max=stats.r_max,
        atomic_numbers=stats.atomic_numbers,
    ) as dataset:
        torch_loader = torch_geometric.loader.DataLoader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            drop_last=False,
        )
        torch_avg, torch_mean, torch_rms = compute_statistics_torch(
            torch_loader,
            stats.atomic_energies,
            stats.atomic_numbers,
        )

    jax_z_table = AtomicNumberTable(list(stats.atomic_numbers))
    graphs = atoms_to_graphs(train_file, stats.r_max, jax_z_table)
    jax_loader = build_loader(
        graphs,
        batch_size=2,
        shuffle=False,
        max_nodes=None,
        max_edges=None,
    )
    jax_avg, jax_mean, jax_rms = compute_statistics_jax(
        jax_loader,
        stats.atomic_energies,
        stats.atomic_numbers,
    )

    assert np.isclose(jax_avg, torch_avg, rtol=1e-6, atol=1e-6)
    assert np.isclose(jax_mean, torch_mean, rtol=1e-6, atol=1e-6)
    assert np.isclose(jax_rms, torch_rms, rtol=1e-6, atol=1e-6)
