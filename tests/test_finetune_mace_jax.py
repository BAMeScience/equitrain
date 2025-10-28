from __future__ import annotations

import warnings
import numpy as np
import pytest
import torch
import jax
from ase import Atoms
from ase.build import bulk
from jax import numpy as jnp
from mace.data.atomic_data import AtomicData
from mace.data.utils import config_from_atoms
from mace.tools import torch_geometric
from mace.tools.model_script_utils import configure_model as configure_model_torch
from mace.tools.multihead_tools import AtomicNumberTable, prepare_default_head
from mace.tools.scripts_utils import extract_config_mace_model
from mace.tools.torch_geometric.batch import Batch
from mace_jax.cli import mace_torch2jax


def _build_structures() -> list[Atoms]:
    structures: list[Atoms] = []

    base = bulk('NaCl', 'rocksalt', a=5.64).repeat((1, 1, 1))
    structures.append(base)

    displaced = base.copy()
    displaced.positions += 0.05 * np.random.default_rng(seed=0).normal(
        size=displaced.positions.shape
    )
    structures.append(displaced)

    return structures


def _build_statistics(zs: list[int]) -> dict:
    return {
        'mean': [0.0],
        'std': [1.0],
        'avg_num_neighbors': 4.0,
        'r_max': 3.5,
        'atomic_numbers': AtomicNumberTable(zs),
        'atomic_energies': [0.0 for _ in zs],
    }


def _create_args(statistics: dict) -> object:
    from mace.tools import build_default_arg_parser, check_args

    args_list = [
        '--name',
        'equitrain-test',
        '--interaction_first',
        'RealAgnosticInteractionBlock',
        '--interaction',
        'RealAgnosticResidualInteractionBlock',
        '--num_channels',
        '8',
        '--max_L',
        '1',
        '--max_ell',
        '1',
        '--num_interactions',
        '1',
        '--correlation',
        '1',
        '--num_radial_basis',
        '4',
        '--num_cutoff_basis',
        '4',
        '--MLP_irreps',
        '8x0e',
        '--distance_transform',
        'Agnesi',
        '--pair_repulsion',
    ]

    args = build_default_arg_parser().parse_args(args_list)
    args, _ = check_args(args)

    args.mean = statistics['mean']
    args.std = statistics['std']
    args.compute_energy = True
    args.compute_forces = False
    args.compute_dipole = False
    args.compute_polarizability = False
    args.loss = 'energy'
    args.device = 'cpu'
    args.train_file = ''
    args.valid_file = ''
    args.test_file = ''
    args.test_dir = ''
    args.E0s = None
    args.statistics_file = None
    args.key_specification = None
    args.valid_fraction = None
    args.config_type_weights = None
    args.keep_isolated_atoms = False
    args.heads = prepare_default_head(args)
    args.avg_num_neighbors = statistics['avg_num_neighbors']
    args.r_max = statistics['r_max']
    args.only_cueq = False
    args.apply_cutoff = True
    args.use_reduced_cg = False
    args.use_so3 = False
    args.embedding_specs = None
    args.use_embedding_readout = False
    args.use_last_readout_only = False
    args.use_agnostic_product = False
    args.compute_stress = True

    return args


def _batch_to_jax(batch: Batch) -> dict[str, jnp.ndarray]:
    converted: dict[str, jnp.ndarray] = {}
    for key in batch.keys:
        value = batch[key]
        if isinstance(value, torch.Tensor):
            converted[key] = jnp.asarray(value.detach().cpu().numpy())
        else:
            converted[key] = value
    return converted


@pytest.mark.skipif(torch.cuda.is_available(), reason='CPU-only reference test')
def test_finetune_mace_jax_matches_torch():
    pytest.importorskip('mace')
    pytest.importorskip('mace_jax')

    structures = _build_structures()
    zs = sorted({int(z) for atoms in structures for z in atoms.get_atomic_numbers()})
    statistics = _build_statistics(zs)

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
    batch_jax = _batch_to_jax(batch)

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

    torch_energy = torch_output['energy'].detach().cpu().numpy()
    torch_forces = torch_output['forces'].detach().cpu().numpy()
    torch_stress = torch_output['stress'].detach().cpu().numpy()

    config = extract_config_mace_model(torch_model)
    config['atomic_energies'] = statistics['atomic_energies']
    config['atomic_numbers'] = statistics['atomic_numbers'].zs

    jax_model, jax_variables, _ = mace_torch2jax.convert_model(torch_model, config)
    jax_output = jax_model.apply(jax_variables, batch_jax, compute_stress=True)

    energy_jax = np.asarray(jax_output['energy'])
    forces_jax = np.asarray(jax_output['forces'])
    stress_jax = np.asarray(jax_output['stress'])

    np.testing.assert_allclose(energy_jax, torch_energy, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(forces_jax, torch_forces, rtol=1e-6, atol=2e-6)
    np.testing.assert_allclose(stress_jax, torch_stress, rtol=1e-6, atol=1e-6)
