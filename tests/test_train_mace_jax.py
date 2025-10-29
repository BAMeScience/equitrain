from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import jraph
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from flax import serialization
from mace.data.atomic_data import AtomicData
from mace.data.utils import config_from_atoms
from mace.tools import torch_geometric
from mace.tools.scripts_utils import extract_config_mace_model
from mace_jax.cli import mace_torch2jax
from mace_jax.data.utils import (
    AtomicNumberTable as JaxAtomicNumberTable,
    Configuration as JaxConfiguration,
    graph_from_configuration,
)

from equitrain import get_args_parser_train, train as equitrain_train
from equitrain.backends.jax_utils import DEFAULT_CONFIG_NAME, DEFAULT_PARAMS_NAME
from equitrain.data.format_hdf5.dataset import HDF5Dataset
from equitrain.data.backend_jax.atoms_to_graphs import graph_to_data
from equitrain.utility_test import MaceWrapper as TorchMaceWrapper


def _build_structures() -> list[Atoms]:
    base = bulk('NaCl', 'rocksalt', a=5.0).repeat((1, 1, 1))
    displaced = base.copy()
    displaced.positions += 0.05 * np.random.default_rng(seed=0).normal(
        size=displaced.positions.shape
    )
    return [base, displaced]


def _write_dataset(path: Path, structures: list[Atoms]) -> None:
    dataset = HDF5Dataset(path, mode='w')
    try:
        for idx, atoms in enumerate(structures):
            atoms_copy = atoms.copy()
            num_atoms = len(atoms_copy)
            zeros_forces = np.zeros((num_atoms, 3), dtype=np.float64)
            stress = np.zeros(6, dtype=np.float64)
            atoms_copy.calc = SinglePointCalculator(
                atoms_copy,
                energy=0.0,
                forces=zeros_forces,
                stress=stress,
                dipole=np.zeros(3, dtype=np.float64),
            )
            atoms_copy.info['virials'] = np.zeros((3, 3), dtype=np.float64)
            atoms_copy.info['dipole'] = np.zeros(3, dtype=np.float64)
            atoms_copy.info['energy_weight'] = 1.0
            atoms_copy.info['forces_weight'] = 0.0
            atoms_copy.info['stress_weight'] = 0.0
            atoms_copy.info['virials_weight'] = 0.0
            atoms_copy.info['dipole_weight'] = 0.0
            dataset[idx] = atoms_copy
    finally:
        dataset.close()


def _make_torch_batch(structures: list[Atoms], wrapper: TorchMaceWrapper):
    data_list = []
    for atoms in structures:
        config = config_from_atoms(atoms)
        config.pbc = [bool(x) for x in config.pbc]
        data_list.append(
            AtomicData.from_config(
                config,
                z_table=wrapper.atomic_numbers,
                cutoff=float(wrapper.r_max),
            )
        )
    batch = torch_geometric.batch.Batch.from_data_list(data_list)
    for key in batch.keys:
        value = getattr(batch, key)
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            setattr(batch, key, value.float())
    if getattr(batch, 'y', None) is None:
        num_graphs = batch.ptr.numel() - 1 if hasattr(batch, 'ptr') else len(data_list)
        batch.y = torch.zeros(num_graphs, dtype=batch.positions.dtype)
    return batch


def _make_jax_graph(structures: list[Atoms], wrapper: TorchMaceWrapper):
    z_table = JaxAtomicNumberTable(tuple(int(z) for z in list(wrapper.atomic_numbers)))
    graphs = []
    for atoms in structures:
        config = JaxConfiguration(
            atomic_numbers=np.asarray(atoms.get_atomic_numbers(), dtype=np.int32),
            positions=np.asarray(atoms.positions, dtype=np.float32),
            energy=np.array(0.0, dtype=np.float32),
            forces=np.zeros((len(atoms), 3), dtype=np.float32),
            stress=np.zeros((3, 3), dtype=np.float32),
            cell=np.asarray(atoms.cell.array, dtype=np.float32),
            pbc=tuple(bool(x) for x in atoms.pbc),
            weight=np.array(1.0, dtype=np.float32),
        )
        graphs.append(
            graph_from_configuration(
                config,
                cutoff=float(wrapper.r_max),
                z_table=z_table,
            )
        )
    return jraph.batch_np(graphs)


def _sanitize_config(config: dict) -> dict:
    def _convert(value):
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        module = getattr(value, '__module__', '')
        if module.startswith(('e3nn', 'e3nn_jax')):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        if isinstance(value, (bool, int, float, str)) or value is None:
            return value
        if isinstance(value, np.ndarray):
            return value.tolist()
        if hasattr(value, 'tolist'):
            return value.tolist()
        if isinstance(value, type):
            return value.__name__
        return str(value)

    return {key: _convert(val) for key, val in config.items()}


def _export_jax_model(
    torch_model_path: Path,
    atomic_numbers: list[int],
    atomic_energies: list[float],
    r_max: float,
    target_dir: Path,
):
    torch_model = torch.load(torch_model_path, weights_only=False).float().eval()
    config = extract_config_mace_model(torch_model)
    config['atomic_numbers'] = [int(z) for z in atomic_numbers]
    config['atomic_energies'] = [float(x) for x in atomic_energies]
    config['r_max'] = float(r_max)

    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / DEFAULT_CONFIG_NAME).write_text(json.dumps(_sanitize_config(config)))

    jax_module, jax_params, _ = mace_torch2jax.convert_model(torch_model, config)
    (target_dir / DEFAULT_PARAMS_NAME).write_bytes(serialization.to_bytes(jax_params))
    return jax_module, jax_params


@pytest.mark.skipif(torch.cuda.is_available(), reason='CPU-only reference test')
def test_finetune_torch_and_jax_match(tmp_path, mace_model_path):
    pytest.importorskip('mace')
    pytest.importorskip('mace_jax')

    structures = _build_structures()
    train_file = tmp_path / 'train.h5'
    valid_file = tmp_path / 'valid.h5'
    _write_dataset(train_file, structures)
    _write_dataset(valid_file, structures)

    args_torch = get_args_parser_train().parse_args([])
    args_torch.backend = 'torch'
    args_torch.train_file = str(train_file)
    args_torch.valid_file = str(valid_file)
    args_torch.test_file = None
    args_torch.output_dir = str(tmp_path / 'torch_out')
    args_torch.model = TorchMaceWrapper(args_torch, filename_model=mace_model_path)
    args_torch.epochs = 1
    args_torch.train_max_steps = 2
    args_torch.valid_max_steps = 1
    args_torch.batch_size = 1
    args_torch.lr = 1e-4
    args_torch.weight_decay = 0.0
    args_torch.opt = 'sgd'
    args_torch.momentum = 0.0
    args_torch.energy_weight = 1.0
    args_torch.forces_weight = 0.0
    args_torch.stress_weight = 0.0
    args_torch.shuffle = False
    args_torch.scheduler = 'step'
    args_torch.gamma = 1.0
    args_torch.step_size = 1
    args_torch.workers = 0
    args_torch.pin_memory = False
    args_torch.tqdm = False
    args_torch.verbose = 0

    equitrain_train(args_torch)
    torch_model = args_torch.model.float().eval()

    atomic_numbers = [int(z) for z in list(torch_model.atomic_numbers)]
    atomic_energies = list(torch_model.atomic_energies)
    r_max = torch_model.r_max

    jax_model_dir = tmp_path / 'jax_model'
    jax_module, jax_template_params = _export_jax_model(
        Path(mace_model_path),
        atomic_numbers,
        atomic_energies,
        r_max,
        jax_model_dir,
    )

    args_jax = get_args_parser_train().parse_args([])
    args_jax.backend = 'jax'
    args_jax.model = str(jax_model_dir)
    args_jax.train_file = str(train_file)
    args_jax.valid_file = str(valid_file)
    args_jax.test_file = None
    args_jax.output_dir = str(tmp_path / 'jax_out')
    args_jax.epochs = 1
    args_jax.train_max_steps = 2
    args_jax.valid_max_steps = 1
    args_jax.batch_size = 1
    args_jax.lr = 1e-4
    args_jax.weight_decay = 0.0
    args_jax.opt = 'sgd'
    args_jax.energy_weight = 1.0
    args_jax.forces_weight = 0.0
    args_jax.stress_weight = 0.0
    args_jax.scheduler = 'constant'
    args_jax.shuffle = False
    args_jax.workers = 0
    args_jax.pin_memory = False
    args_jax.tqdm = False
    args_jax.verbose = 0
    args_jax.dtype = 'float32'

    equitrain_train(args_jax)

    jax_params_path = Path(args_jax.output_dir) / 'jax_params.msgpack'
    raw_state = serialization.msgpack_restore(jax_params_path.read_bytes())
    template_state = serialization.to_state_dict(jax_template_params)
    if 'interactions' not in raw_state['params'] and 'interactions' in template_state['params']:
        raw_state['params']['interactions'] = template_state['params']['interactions']
    jax_trained_params = serialization.from_state_dict(jax_template_params, raw_state)

    eval_batch = _make_torch_batch(structures, torch_model)
    torch_output = torch_model(eval_batch)
    torch_energy = torch_output['energy'].detach().cpu().numpy()

    graphs = _make_jax_graph(structures, torch_model)
    data_dict = graph_to_data(graphs, num_species=len(atomic_numbers))
    jax_energy = jax_module.apply(
        jax_trained_params,
        data_dict,
        compute_force=False,
        compute_stress=False,
    )
    energy_jax = np.asarray(jax_energy['energy'])

    assert not np.isnan(energy_jax).any(), 'JAX training produced NaN predictions'
    assert not np.isnan(torch_energy).any(), 'Torch training produced NaN predictions'

    np.testing.assert_allclose(energy_jax, torch_energy, rtol=1e-6, atol=1e-6)
