from __future__ import annotations

import warnings
import numpy as np
import pytest
import torch
import jax
from flax import serialization
from mace.data.atomic_data import AtomicData
from mace.data.utils import config_from_atoms
from mace.tools import torch_geometric
from mace.tools.scripts_utils import extract_config_mace_model
from mace_jax.cli import mace_torch2jax

from .conftest import _build_structures, _build_statistics, _create_args


def _make_batch(statistics: dict) -> torch_geometric.batch.Batch:
    structures = _build_structures()
    atomic_data_list = []
    for atoms in structures:
        config = config_from_atoms(atoms)
        config.pbc = [bool(x) for x in config.pbc]
        atomic_data_list.append(
            AtomicData.from_config(
                config,
                z_table=statistics['atomic_numbers'],
                cutoff=float(statistics['r_max']),
            )
        )

    batch = torch_geometric.batch.Batch.from_data_list(atomic_data_list)
    for key in batch.keys:
        value = getattr(batch, key)
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            setattr(batch, key, value.float())
    return batch


def _batch_to_jax(batch: torch_geometric.batch.Batch) -> dict[str, jax.Array]:
    converted: dict[str, jax.Array] = {}
    for key in batch.keys:
        value = getattr(batch, key)
        if isinstance(value, torch.Tensor):
            converted[key] = jax.device_put(value.detach().cpu().numpy())
        else:
            converted[key] = value
    return converted


@pytest.mark.skipif(torch.cuda.is_available(), reason='CPU-only reference test')
def test_small_mace_conversion_matches():
    pytest.importorskip('mace')
    pytest.importorskip('mace_jax')

    statistics = _build_statistics([11, 17])
    batch = _make_batch(statistics)
    if hasattr(batch, 'positions') and isinstance(batch.positions, torch.Tensor):
        batch.positions.requires_grad_(True)
    if hasattr(batch, 'cell') and isinstance(batch.cell, torch.Tensor):
        batch.cell.requires_grad_(True)

    from mace.tools.model_script_utils import configure_model as configure_model_torch

    args = _create_args(statistics)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'.*TorchScript type system doesn\'t support instance-level annotations.*',
            category=UserWarning,
        )
        torch_model, _ = configure_model_torch(
            args,
            train_loader=[],
            atomic_energies=statistics['atomic_energies'],
            heads=args.heads,
            z_table=statistics['atomic_numbers'],
        )
    torch_model = torch_model.float().eval()

    torch_output = torch_model(batch, compute_stress=True)

    config = extract_config_mace_model(torch_model)
    config['atomic_energies'] = statistics['atomic_energies']
    config['atomic_numbers'] = statistics['atomic_numbers'].zs

    jax_model, jax_variables, _ = mace_torch2jax.convert_model(torch_model, config)
    batch_jax = _batch_to_jax(batch)
    jax_output = jax_model.apply(jax_variables, batch_jax, compute_stress=True)

    energy_torch = torch_output['energy'].detach().cpu().numpy()
    forces_torch = torch_output['forces'].detach().cpu().numpy()
    stress_torch = torch_output['stress'].detach().cpu().numpy()

    energy_jax = np.asarray(jax_output['energy'])
    forces_jax = np.asarray(jax_output['forces'])
    stress_jax = np.asarray(jax_output['stress'])

    np.testing.assert_allclose(energy_jax, energy_torch, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(forces_jax, forces_torch, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(stress_jax, stress_torch, rtol=1e-6, atol=1e-6)
