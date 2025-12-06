from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator

pytest.importorskip('mace', reason='MACE is required for MACE JAX integration tests.')
pytest.importorskip('mace_jax', reason='MACE JAX is required for these tests.')
pytest.importorskip('jax', reason='JAX runtime is required for these tests.')

import jax.numpy as jnp  # noqa: E402
import jraph  # noqa: E402
from flax import core as flax_core  # noqa: E402
from flax import serialization, traverse_util  # noqa: E402
from mace.data.atomic_data import AtomicData  # noqa: E402
from mace.data.utils import config_from_atoms  # noqa: E402
from mace.tools import torch_geometric  # noqa: E402
from mace.tools.scripts_utils import extract_config_mace_model  # noqa: E402
from mace_jax.cli import mace_torch2jax  # noqa: E402

from equitrain import get_args_parser_train  # noqa: E402
from equitrain import train as equitrain_train  # noqa: E402
from equitrain.backends.jax_utils import (  # noqa: E402
    DEFAULT_CONFIG_NAME,
    DEFAULT_PARAMS_NAME,
    ModelBundle,
    load_model_bundle,
)
from equitrain.data.atomic import AtomicNumberTable  # noqa: E402
from equitrain.data.backend_jax import (  # noqa: E402
    graph_from_configuration,
    graph_to_data,
)
from equitrain.data.configuration import Configuration as EqConfiguration  # noqa: E402
from equitrain.data.format_hdf5.dataset import HDF5Dataset  # noqa: E402
from equitrain.utility_test import MaceWrapper as TorchMaceWrapper  # noqa: E402
from equitrain.utility_test.mace_support import get_mace_model_path  # noqa: E402


def _cleanup_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


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


def _set_jax_batch_limits(args, nodes=2048, edges=8192):
    args.batch_max_nodes = nodes
    args.batch_max_edges = edges
    return args


def _create_jax_bundle(tmp_path: Path) -> Path:
    args = get_args_parser_train().parse_args([])
    mace_model_path = get_mace_model_path()
    torch_wrapper = TorchMaceWrapper(args, filename_model=mace_model_path)
    torch_model = torch_wrapper.float().eval()
    torch_model_path = tmp_path / 'torch_reference.model'
    torch.save(torch_model.model, torch_model_path)

    atomic_numbers = [int(z) for z in list(torch_wrapper.atomic_numbers)]
    atomic_energies = list(torch_wrapper.atomic_energies)
    r_max = torch_wrapper.r_max

    jax_model_dir = tmp_path / 'jax_bundle'
    _export_jax_model(
        torch_model_path,
        atomic_numbers,
        atomic_energies,
        r_max,
        jax_model_dir,
    )
    return jax_model_dir


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
    z_table = AtomicNumberTable(list(wrapper.atomic_numbers))
    graphs = []
    for atoms in structures:
        num_atoms = len(atoms)
        config = EqConfiguration(
            atomic_numbers=np.asarray(atoms.get_atomic_numbers(), dtype=np.int32),
            positions=np.asarray(atoms.positions, dtype=np.float32),
            energy=0.0,
            forces=np.zeros((num_atoms, 3), dtype=np.float32),
            stress=np.zeros(6, dtype=np.float32),
            cell=np.asarray(atoms.cell.array, dtype=np.float32),
            pbc=tuple(bool(x) for x in atoms.pbc),
            energy_weight=1.0,
            forces_weight=0.0,
            stress_weight=0.0,
            virials=np.zeros((3, 3), dtype=np.float32),
            virials_weight=0.0,
            dipole=np.zeros(3, dtype=np.float32),
            dipole_weight=0.0,
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
        if isinstance(value, list | tuple):
            return [_convert(v) for v in value]
        if isinstance(value, bool | int | float | str) or value is None:
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


def _predict_jax_energy_from_bundle(
    bundle: ModelBundle,
    structures: list[Atoms],
    reference_wrapper: TorchMaceWrapper,
    atomic_numbers: list[int],
    *,
    params=None,
):
    graphs = _make_jax_graph(structures, reference_wrapper)
    data_dict = graph_to_data(graphs, num_species=len(atomic_numbers))
    active_params = bundle.params if params is None else params
    outputs = bundle.module.apply(
        active_params,
        data_dict,
        compute_force=False,
        compute_stress=False,
    )
    return np.asarray(outputs['energy'])


@pytest.mark.skipif(torch.cuda.is_available(), reason='CPU-only reference test')
def test_train_torch_and_jax_match(tmp_path):
    pytest.importorskip('mace')
    pytest.importorskip('mace_jax')

    cleanup_paths: list[Path] = []
    try:
        structures = _build_structures()
        train_file = tmp_path / 'train.h5'
        valid_file = tmp_path / 'valid.h5'
        _write_dataset(train_file, structures)
        _write_dataset(valid_file, structures)

        args_torch = get_args_parser_train().parse_args([])
        args_torch.backend = 'torch'
        args_torch.train_file = str(train_file)
        args_torch.valid_file = str(valid_file)
        args_torch.test_file = str(valid_file)
        args_torch.output_dir = str(tmp_path / 'torch_out')
        mace_model_path = get_mace_model_path()
        args_torch.model = TorchMaceWrapper(args_torch, filename_model=mace_model_path)
        args_torch.epochs = 1
        args_torch.train_max_steps = 2
        args_torch.valid_max_steps = 1
        args_torch.batch_size = 1
        args_torch.lr = 5e-3
        args_torch.weight_decay = 0.0
        args_torch.momentum = 0.0
        args_torch.energy_weight = 1.0
        args_torch.forces_weight = 0.0
        args_torch.stress_weight = 0.0
        args_torch.shuffle = False
        args_torch.scheduler = 'step'
        args_torch.gamma = 1.0
        args_torch.step_size = 1
        args_torch.num_workers = 0
        args_torch.pin_memory = False
        args_torch.tqdm = False
        args_torch.verbose = 0
        args_torch.dtype = 'float32'
        args_torch.seed = 123
        args_torch.gradient_clipping = 5e-4
        args_torch.ema = True
        args_torch.ema_decay = 0.5

        torch_model_pre = args_torch.model.float().eval()
        batch = _make_torch_batch(structures, torch_model_pre)
        torch_energy_pre = torch_model_pre(batch)['energy'].detach().cpu().numpy()
        torch_model_path = tmp_path / 'torch_pre.model'
        torch.save(torch_model_pre.model, torch_model_path)

        equitrain_train(args_torch)
        cleanup_paths.append(Path(args_torch.output_dir))
        torch_model = args_torch.model.float().eval()
        torch_energy_post = (
            torch_model(_make_torch_batch(structures, torch_model))['energy']
            .detach()
            .cpu()
            .numpy()
        )

        atomic_numbers = [int(z) for z in list(torch_model_pre.atomic_numbers)]
        atomic_energies = list(torch_model_pre.atomic_energies)
        r_max = torch_model_pre.r_max

        jax_model_dir = tmp_path / 'jax_model'
        _export_jax_model(
            torch_model_path,
            atomic_numbers,
            atomic_energies,
            r_max,
            jax_model_dir,
        )
        cleanup_paths.append(jax_model_dir)

        bundle = load_model_bundle(str(jax_model_dir), dtype='float32')
        jax_energy_pre = _predict_jax_energy_from_bundle(
            bundle,
            structures,
            torch_model_pre,
            atomic_numbers,
        )
        np.testing.assert_allclose(
            jax_energy_pre,
            torch_energy_pre,
            rtol=1e-5,
            atol=1e-4,
            err_msg='Torch and JAX predictions differ before training.',
        )

        args_jax = get_args_parser_train().parse_args([])
        args_jax.backend = 'jax'
        args_jax.model = str(jax_model_dir)
        args_jax.train_file = str(train_file)
        args_jax.valid_file = str(valid_file)
        args_jax.test_file = str(valid_file)
        args_jax.output_dir = str(tmp_path / 'jax_out')
        args_jax.epochs = 1
        args_jax.train_max_steps = 2
        args_jax.valid_max_steps = 1
        args_jax.batch_size = 1
        _set_jax_batch_limits(args_jax)
        args_jax.lr = 5e-3
        args_jax.weight_decay = 0.0
        args_jax.energy_weight = 1.0
        args_jax.forces_weight = 0.0
        args_jax.stress_weight = 0.0
        args_jax.scheduler = 'constant'
        args_jax.shuffle = False
        args_jax.num_workers = 0
        args_jax.pin_memory = False
        args_jax.tqdm = False
        args_jax.verbose = 0
        args_jax.dtype = 'float32'
        args_jax.seed = 123
        args_jax.gradient_clipping = 5e-4
        args_jax.ema = True
        args_jax.ema_decay = 0.5
        jax_training_summary = equitrain_train(args_jax)
        cleanup_paths.append(Path(args_jax.output_dir))
        assert jax_training_summary is not None, (
            'JAX backend must return training summary'
        )
        for key in ('train_loss', 'val_loss', 'test_loss'):
            value = jax_training_summary.get(key)
            assert value is not None, f'JAX summary missing {key}'
            assert np.isfinite(value), f'JAX summary {key} not finite: {value}'

        jax_params_path = Path(args_jax.output_dir) / 'jax_params.msgpack'
        raw_state = serialization.msgpack_restore(jax_params_path.read_bytes())
        if 'params' not in raw_state:
            raw_state = {'params': raw_state}
        loaded_params_state = flax_core.unfreeze(raw_state['params'])

        template_state = flax_core.unfreeze(bundle.params)
        template_params_state = copy.deepcopy(template_state['params'])

        if 'delta' in loaded_params_state:
            delta_state = loaded_params_state.pop('delta')
            flat_base = traverse_util.flatten_dict(template_params_state)
            flat_delta = traverse_util.flatten_dict(delta_state)
            combined = {}
            for key, base_val in flat_base.items():
                delta_val = flat_delta.get(key)
                if delta_val is None:
                    combined[key] = base_val
                else:
                    combined[key] = jnp.asarray(base_val) + jnp.asarray(delta_val)
            merged_params_state = traverse_util.unflatten_dict(combined)
        else:

            def _merge(base_dict, updates):
                for key, value in updates.items():
                    if isinstance(value, dict):
                        base_dict[key] = _merge(dict(base_dict.get(key, {})), value)
                    else:
                        base_dict[key] = value
                return base_dict

            merged_params_state = _merge(template_params_state, loaded_params_state)

        base_variables = flax_core.unfreeze(bundle.params)
        base_variables['params'] = merged_params_state
        final_variables = flax_core.freeze(base_variables)

        jax_energy_post = _predict_jax_energy_from_bundle(
            bundle,
            structures,
            torch_model_pre,
            atomic_numbers,
            params=final_variables,
        )

        assert not np.isnan(jax_energy_post).any(), (
            'JAX training produced NaN predictions'
        )
        assert not np.isnan(torch_energy_post).any(), (
            'Torch training produced NaN predictions'
        )

        gap_pre = np.max(np.abs(jax_energy_pre - torch_energy_pre))
        gap_post = np.max(np.abs(jax_energy_post - torch_energy_post))

        assert gap_pre <= 1e-4, f'Pre-training gap too large: {gap_pre:.6f}'
        assert gap_post <= 1e-4, f'Post-training gap too large: {gap_post:.6f}'

        assert jax_training_summary['initial_val_loss'] is not None
        assert len(jax_training_summary['lr_history']) == args_jax.epochs + 1
        assert jax_training_summary['best_epoch'] is not None
    finally:
        for path in cleanup_paths:
            _cleanup_path(path)


def test_jax_weighted_sampler_not_supported(tmp_path):
    pytest.importorskip('mace_jax')

    jax_bundle = _create_jax_bundle(tmp_path)
    data_dir = Path(__file__).with_name('data')

    args = get_args_parser_train().parse_args([])
    args.backend = 'jax'
    args.model = str(jax_bundle)
    args.train_file = str(data_dir / 'train.h5')
    args.valid_file = str(data_dir / 'valid.h5')
    args.test_file = None
    args.output_dir = str(tmp_path / 'out_weighted_sampler')
    args.epochs = 1
    args.batch_size = 1
    _set_jax_batch_limits(args)
    args.lr = 1e-3
    args.weighted_sampler = True

    with pytest.raises(ValueError, match='does not support weighted'):
        equitrain_train(args)


def test_jax_step_scheduler_updates_learning_rate(tmp_path):
    pytest.importorskip('mace_jax')

    jax_bundle = _create_jax_bundle(tmp_path)
    data_dir = Path(__file__).with_name('data')

    args = get_args_parser_train().parse_args([])
    args.backend = 'jax'
    args.model = str(jax_bundle)
    args.train_file = str(data_dir / 'train.h5')
    args.valid_file = str(data_dir / 'valid.h5')
    args.test_file = None
    args.output_dir = str(tmp_path / 'out_step_scheduler')
    args.epochs = 2
    args.batch_size = 1
    _set_jax_batch_limits(args)
    args.lr = 1e-3
    args.scheduler = 'step'
    args.gamma = 0.5
    args.step_size = 1
    args.train_max_steps = 1
    args.valid_max_steps = 1
    args.verbose = 0
    args.tqdm = False

    summary = equitrain_train(args)
    assert len(summary['lr_history']) == args.epochs + 1
    assert summary['lr_history'][0] == pytest.approx(args.lr)
    assert summary['lr_history'][1] == pytest.approx(args.lr * args.gamma)
    assert summary['lr_history'][2] == pytest.approx(args.lr * args.gamma * args.gamma)


def test_jax_plateau_scheduler_reduces_learning_rate(tmp_path):
    pytest.importorskip('mace_jax')

    jax_bundle = _create_jax_bundle(tmp_path)
    data_dir = Path(__file__).with_name('data')

    args = get_args_parser_train().parse_args([])
    args.backend = 'jax'
    args.model = str(jax_bundle)
    args.train_file = str(data_dir / 'train.h5')
    args.valid_file = str(data_dir / 'valid.h5')
    args.test_file = None
    args.output_dir = str(tmp_path / 'out_plateau_scheduler')
    args.epochs = 2
    args.batch_size = 1
    _set_jax_batch_limits(args)
    args.lr = 1e-3
    args.scheduler = 'plateau'
    args.plateau_factor = 0.5
    args.plateau_patience = 0
    args.plateau_mode = 'min'
    args.plateau_threshold = 0.0
    args.plateau_threshold_mode = 'rel'
    args.plateau_eps = 0.0
    args.train_max_steps = 1
    args.valid_max_steps = 1
    args.verbose = 0
    args.tqdm = False

    summary = equitrain_train(args)
    assert len(summary['lr_history']) == args.epochs + 1
    assert summary['lr_history'][0] == pytest.approx(args.lr)
    assert summary['lr_history'][1] == pytest.approx(args.lr * args.plateau_factor)
    assert summary['lr_history'][2] == pytest.approx(
        args.lr * args.plateau_factor * args.plateau_factor
    )
