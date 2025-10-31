from __future__ import annotations

import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pytest
import torch
from ase import Atoms
from flax import core as flax_core
from flax import serialization
from mace.data.atomic_data import AtomicData
from mace.data.utils import config_from_atoms
from mace.tools import torch_geometric
from mace.tools.scripts_utils import extract_config_mace_model
from mace_jax.cli import mace_torch2jax
from mace_jax.data.utils import AtomicNumberTable as JaxAtomicNumberTable
from mace_jax.data.utils import Configuration as JaxConfiguration
from mace_jax.data.utils import graph_from_configuration
from torch.serialization import add_safe_globals

from equitrain import get_args_parser_train
from equitrain import train as equitrain_train
from equitrain.backends.jax_utils import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_PARAMS_NAME,
    load_model_bundle,
)
from equitrain.data.backend_jax.atoms_to_graphs import graph_to_data
from equitrain.data.format_hdf5.dataset import HDF5Dataset
from tests.test_finetune_mace import FinetuneMaceWrapper as TorchFinetuneWrapper

add_safe_globals([slice])


def _load_structures(path: Path) -> list[Atoms]:
    dataset = HDF5Dataset(path, mode='r')
    try:
        return [dataset[idx] for idx in range(len(dataset))]
    finally:
        dataset.close()


def _make_torch_batch(structures: list[Atoms], wrapper: TorchFinetuneWrapper):
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


def _make_jax_graph(structures: list[Atoms], wrapper: TorchFinetuneWrapper):
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
        if callable(value):
            return getattr(value, '__name__', str(value))
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


def _convert_torch_model_to_jax_params(torch_model, atomic_numbers: list[int]):
    torch_model = torch_model.float().eval()
    config = extract_config_mace_model(torch_model)
    config['atomic_numbers'] = [int(z) for z in atomic_numbers]
    config['atomic_energies'] = [
        float(x) for x in torch_model.atomic_energies_fn.atomic_energies.detach().cpu()
    ]
    config['r_max'] = float(torch_model.r_max.item())
    _, params, _ = mace_torch2jax.convert_model(torch_model, config)
    return flax_core.freeze(params)


@pytest.mark.skipif(torch.cuda.is_available(), reason='CPU-only reference test')
def test_finetune_mace_jax(tmp_path, mace_model_path):
    pytest.importorskip('mace')
    pytest.importorskip('mace_jax')

    data_dir = Path(__file__).with_name('data')
    train_file = data_dir / 'train.h5'
    valid_file = data_dir / 'valid.h5'
    structures = _load_structures(train_file)

    args_torch = get_args_parser_train().parse_args([])
    args_torch.backend = 'torch'
    args_torch.train_file = str(train_file)
    args_torch.valid_file = str(valid_file)
    args_torch.test_file = None
    args_torch.output_dir = str(tmp_path / 'torch_out')
    args_torch.epochs = 1
    args_torch.train_max_steps = 2
    args_torch.valid_max_steps = 1
    args_torch.batch_size = 1
    args_torch.opt = 'momentum'
    args_torch.lr = 1e-4
    args_torch.weight_decay = 0.0
    args_torch.momentum = 0.0
    args_torch.scheduler = 'step'
    args_torch.gamma = 1.0
    args_torch.step_size = 1
    args_torch.shuffle = False
    args_torch.workers = 0
    args_torch.pin_memory = False
    args_torch.tqdm = False
    args_torch.verbose = 0
    args_torch.dtype = 'float32'
    args_torch.energy_weight = 1.0
    args_torch.forces_weight = 0.0
    args_torch.stress_weight = 0.0
    args_torch.model = TorchFinetuneWrapper(args_torch, filename_model=mace_model_path)

    torch_model_pre = args_torch.model.float().eval()
    torch_batch = _make_torch_batch(structures, torch_model_pre)
    with torch.no_grad():
        torch_energy_pre = torch_model_pre(torch_batch)['energy'].detach().cpu().numpy()

    equitrain_train(args_torch)

    torch_model_post = args_torch.model.float().eval()
    with torch.no_grad():
        torch_energy_post = (
            torch_model_post(_make_torch_batch(structures, torch_model_post))['energy']
            .detach()
            .cpu()
            .numpy()
        )

    torch_atomic_pre = (
        torch_model_pre.model.atomic_energies_fn.atomic_energies.detach().cpu().clone()
    )
    torch_atomic_post = (
        torch_model_post.model.atomic_energies_fn.atomic_energies.detach().cpu()
    )
    np.testing.assert_allclose(
        torch_atomic_post.numpy(),
        torch_atomic_pre.numpy(),
        atol=0.0,
        err_msg='Torch fine-tuning modified atomic energies.',
    )

    torch_pre_path = tmp_path / 'torch_pre.model'
    torch_post_path = tmp_path / 'torch_finetuned.model'
    torch.save(torch_model_pre.model, torch_pre_path)
    args_torch.model.export(str(torch_post_path))

    atomic_numbers = [int(z) for z in list(torch_model_pre.atomic_numbers)]
    atomic_energies = list(torch_model_pre.atomic_energies)
    r_max = torch_model_pre.r_max

    jax_model_dir = tmp_path / 'jax_model'
    _export_jax_model(
        torch_pre_path,
        atomic_numbers,
        atomic_energies,
        r_max,
        jax_model_dir,
    )

    bundle = load_model_bundle(str(jax_model_dir), dtype='float32')
    graphs = _make_jax_graph(structures, torch_model_pre)
    data_dict = graph_to_data(graphs, num_species=len(atomic_numbers))
    jax_energy_pre = np.asarray(
        bundle.module.apply(
            bundle.params,
            data_dict,
            compute_force=False,
            compute_stress=False,
        )['energy']
    )

    np.testing.assert_allclose(
        jax_energy_pre,
        torch_energy_pre,
        rtol=1e-5,
        atol=1e-4,
        err_msg='Torch and JAX predictions differ before fine-tuning.',
    )

    jax_params_from_torch_raw = _convert_torch_model_to_jax_params(
        torch.load(torch_post_path, weights_only=False).float().eval(),
        atomic_numbers,
    )
    target_state = serialization.to_state_dict(jax_params_from_torch_raw)
    base_state = serialization.to_state_dict(bundle.params)
    if (
        'interactions' not in target_state['params']
        and 'interactions' in base_state['params']
    ):
        target_state['params']['interactions'] = base_state['params']['interactions']
    jax_params_from_torch = serialization.from_state_dict(bundle.params, target_state)
    jax_energy_post = np.asarray(
        bundle.module.apply(
            jax_params_from_torch,
            data_dict,
            compute_force=False,
            compute_stress=False,
        )['energy']
    )
    np.testing.assert_allclose(
        jax_energy_post,
        torch_energy_post,
        rtol=1e-5,
        atol=1e-5,
        err_msg='Torch and JAX predictions differ after fine-tuning.',
    )

    jax_atomic_post = np.asarray(
        serialization.to_state_dict(jax_params_from_torch)['params'][
            'atomic_energies_fn'
        ]['atomic_energies']
    )
    np.testing.assert_allclose(
        jax_atomic_post,
        serialization.to_state_dict(bundle.params)['params']['atomic_energies_fn'][
            'atomic_energies'
        ],
        atol=0.0,
        err_msg='JAX fine-tuning modified atomic energies.',
    )
