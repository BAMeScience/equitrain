from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jraph
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from ase import Atoms
from flax import core as flax_core
from flax import serialization
from flax import traverse_util
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
from equitrain.backends.jax_loss_fn import LossSettings, build_loss_fn
from equitrain.backends.jax_utils import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_PARAMS_NAME,
    load_model_bundle,
    ModelBundle,
)
from equitrain.backends.jax_wrappers import MaceWrapper as JaxMaceWrapper
from equitrain.data.backend_jax import atoms_to_graphs, build_loader, make_apply_fn
from equitrain.backends.torch_checkpoint import load_model_state as load_torch_model_state
from equitrain.data.backend_jax.atoms_to_graphs import graph_to_data
from equitrain.data.format_hdf5.dataset import HDF5Dataset
from tests.test_finetune_mace import FinetuneMaceWrapper as TorchFinetuneWrapper
from tests.test_train_mace_jax import (
    _build_structures as _build_match_structures,
    _write_dataset as _write_match_dataset,
)

add_safe_globals([slice])


_FINE_TUNE_LR = 2.5e-3
_MAX_STEPS = 24
_DATASET_LIMIT = 16
_PARITY_STEPS = 64


class _DeltaWrapperModule:
    def __init__(self, inner_module):
        self._inner = inner_module

    def init(self, rng, template):
        base_vars = self._inner.init(rng, template)
        base_vars = flax_core.unfreeze(base_vars)
        params_tree = base_vars.pop('params', {})
        delta_tree = jtu.tree_map(lambda x: jnp.zeros_like(x), params_tree)
        base_vars['params'] = {'delta': delta_tree}
        base_vars['base_params'] = params_tree
        return flax_core.freeze(base_vars)

    def apply(self, variables, *args, **kwargs):
        params = variables.get('params', {})
        base_tree = variables.get('base_params', {})
        if base_tree and 'delta' in params:
            base_tree = jtu.tree_map(jax.lax.stop_gradient, base_tree)
            combined = jtu.tree_map(lambda b, d: b + d, base_tree, params['delta'])
            actual_vars = flax_core.freeze({'params': combined})
            return self._inner.apply(actual_vars, *args, **kwargs)
        return self._inner.apply(variables, *args, **kwargs)


def _ensure_delta_params(variables: flax_core.FrozenDict) -> flax_core.FrozenDict:
    unfrozen = flax_core.unfreeze(variables)
    params_tree = unfrozen.get('params')

    if params_tree is None:
        return flax_core.freeze(unfrozen)

    if 'base' in params_tree and 'delta' in params_tree:
        base_tree = params_tree['base']
        delta_tree = params_tree['delta']
        unfrozen['params'] = {'delta': delta_tree}
        unfrozen['base_params'] = base_tree
        return flax_core.freeze(unfrozen)

    if 'delta' in params_tree:
        if 'base_params' not in unfrozen:
            base_shape = jtu.tree_map(lambda x: jnp.zeros_like(x), params_tree['delta'])
            unfrozen['base_params'] = base_shape
        return flax_core.freeze(unfrozen)

    base_tree = params_tree
    delta_tree = jtu.tree_map(lambda x: jnp.zeros_like(x), base_tree)
    unfrozen['params'] = {'delta': delta_tree}
    unfrozen['base_params'] = base_tree
    return flax_core.freeze(unfrozen)


def _init_common_args(args, train_file, valid_file, output_dir, *, lr=_FINE_TUNE_LR, max_steps=_MAX_STEPS):
    args.train_file = str(train_file)
    args.valid_file = str(valid_file)
    args.test_file = None
    args.output_dir = str(output_dir)
    args.epochs = 1
    args.train_max_steps = max_steps
    args.valid_max_steps = max_steps
    args.batch_size = 1
    args.lr = lr
    args.weight_decay = 0.0
    args.momentum = 0.0
    args.shuffle = False
    args.workers = 0
    args.pin_memory = False
    args.tqdm = False
    args.verbose = 0
    args.dtype = 'float32'
    args.energy_weight = 1.0
    args.forces_weight = 0.0
    args.stress_weight = 0.0
    return args


def _build_torch_args(train_file, valid_file, output_dir, mace_model_path, *, max_steps=_MAX_STEPS, lr=_FINE_TUNE_LR):
    args = _init_common_args(
        get_args_parser_train().parse_args([]),
        train_file,
        valid_file,
        output_dir,
        lr=lr,
        max_steps=max_steps,
    )
    args.backend = 'torch'
    args.opt = 'momentum'
    args.scheduler = 'step'
    args.gamma = 1.0
    args.step_size = 1
    args.model = TorchFinetuneWrapper(args, filename_model=mace_model_path)
    return args


def _build_jax_args(train_file, valid_file, output_dir, model_path, *, max_steps=_MAX_STEPS, lr=_FINE_TUNE_LR):
    args = _init_common_args(
        get_args_parser_train().parse_args([]),
        train_file,
        valid_file,
        output_dir,
        lr=lr,
        max_steps=max_steps,
    )
    args.backend = 'jax'
    args.model = str(model_path)
    args.opt = 'momentum'
    args.scheduler = 'constant'
    args.freeze_params = [r'params\.base\..*']
    args.unfreeze_params = [r'params\.delta\..*']
    return args


def _predict_torch_energy(model, structures: list[Atoms]) -> np.ndarray:
    batch = _make_torch_batch(structures, model)
    with torch.no_grad():
        return model(batch)['energy'].detach().cpu().numpy()


def _structures_to_jax_input(structures: list[Atoms], wrapper: TorchFinetuneWrapper):
    graphs = _make_jax_graph(structures, wrapper)
    return graph_to_data(graphs, num_species=len(wrapper.atomic_numbers))


def _predict_jax_energy(bundle, data_dict, *, params=None) -> np.ndarray:
    variables = params if params is not None else bundle.params
    outputs = bundle.module.apply(
        variables,
        data_dict,
        compute_force=False,
        compute_stress=False,
    )
    return np.asarray(outputs['energy'])


def _copy_dataset_subset(src: Path, dst: Path, limit: int) -> Path:
    if dst.exists():
        return dst

    src_dataset = HDF5Dataset(src, mode='r')
    dst_dataset = HDF5Dataset(dst, mode='w')
    try:
        for index in range(min(limit, len(src_dataset))):
            dst_dataset[index] = src_dataset[index]
    finally:
        src_dataset.close()
        dst_dataset.close()

    return dst


@contextmanager
def _patch_jax_loader_for_deltas():
    from equitrain.backends import jax_utils as jax_utils_module

    def patched(model_arg, dtype):
        config_path, params_path = jax_utils_module.resolve_model_paths(model_arg)
        config_data = json.loads(Path(config_path).read_text())

        config_for_build = dict(config_data)
        config_for_build.pop('train_deltas', None)

        jax_utils_module.set_jax_dtype(dtype)

        base_module = mace_torch2jax._build_jax_model(config_for_build)
        wrapped_module = _DeltaWrapperModule(base_module)
        template = mace_torch2jax._prepare_template_data(config_for_build)
        params_bytes = Path(params_path).read_bytes()

        variables_template = wrapped_module.init(jax.random.PRNGKey(0), template)
        try:
            loaded = serialization.from_bytes(variables_template, params_bytes)
        except ValueError:
            base_variables = base_module.init(jax.random.PRNGKey(0), template)
            loaded = serialization.from_bytes(base_variables, params_bytes)

        return ModelBundle(
            config=config_data,
            params=_ensure_delta_params(loaded),
            module=wrapped_module,
        )

    with mock.patch('equitrain.backends.jax_utils.load_model_bundle', patched), \
         mock.patch('equitrain.backends.jax_backend.load_model_bundle', patched), \
         mock.patch('tests.test_finetune_mace_jax.load_model_bundle', patched):
        yield


def _load_structures(path: Path) -> list[Atoms]:
    dataset = HDF5Dataset(path, mode='r')
    try:
        return [dataset[idx] for idx in range(len(dataset))]
    finally:
        dataset.close()


_CKPT_PATTERN = re.compile(r'best_val_epochs@(\d+)_e@([0-9]*\.[0-9]+)')


def _find_best_checkpoint_dir(base_dir: Path) -> Path:
    best_dir: Path | None = None
    best_val: float | None = None
    for candidate in base_dir.glob('best_val_epochs@*_e@*'):
        if not candidate.is_dir():
            continue
        match = _CKPT_PATTERN.match(candidate.name)
        if match is None:
            continue
        val = float(match.group(2))
        if best_val is None or val < best_val:
            best_val = val
            best_dir = candidate
    if best_dir is None:
        raise AssertionError(f'No checkpoint directories found in {base_dir}')
    return best_dir


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
    config['train_deltas'] = True

    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / DEFAULT_CONFIG_NAME).write_text(json.dumps(_sanitize_config(config)))

    jax_module, jax_params, _ = mace_torch2jax.convert_model(torch_model, config)
    (target_dir / DEFAULT_PARAMS_NAME).write_bytes(serialization.to_bytes(jax_params))
    return jax_module, jax_params


def _convert_torch_model_to_jax_params(torch_model, atomic_numbers: list[int], base_params=None):
    torch_model = torch_model.float().eval()
    config = extract_config_mace_model(torch_model)
    config['atomic_numbers'] = [int(z) for z in atomic_numbers]
    config['atomic_energies'] = [
        float(x) for x in torch_model.atomic_energies_fn.atomic_energies.detach().cpu()
    ]
    config['r_max'] = float(torch_model.r_max.item())
    _, params, _ = mace_torch2jax.convert_model(torch_model, config)
    params_tree = flax_core.unfreeze(params.get('params', {}))

    if base_params is not None:
        base_unfrozen = flax_core.unfreeze(base_params)
    else:
        base_unfrozen = {}

    if 'base_params' in base_unfrozen:
        base_tree = flax_core.unfreeze(base_unfrozen['base_params'])
    elif 'params' in base_unfrozen and 'base' in base_unfrozen['params']:
        base_tree = flax_core.unfreeze(base_unfrozen['params']['base'])
    elif base_params is not None:
        raise ValueError('Expected base parameters to include a base tree.')
    else:
        base_tree = params_tree

    flat_params = traverse_util.flatten_dict(params_tree)
    flat_base = traverse_util.flatten_dict(base_tree)

    missing_in_converted = sorted(set(flat_base) - set(flat_params))
    if missing_in_converted:
        missing_keys = ', '.join('.'.join(k) for k in missing_in_converted)
        raise ValueError(f'Converted Torch parameters missing keys: {missing_keys}')

    unexpected_in_converted = sorted(set(flat_params) - set(flat_base))
    if unexpected_in_converted:
        unexpected_keys = ', '.join('.'.join(k) for k in unexpected_in_converted)
        raise ValueError(f'Converted Torch parameters contain unexpected keys: {unexpected_keys}')

    delta_flat = {}
    for key, base_val in flat_base.items():
        new_val = flat_params[key]
        delta_flat[key] = jnp.asarray(new_val) - jnp.asarray(base_val)

    delta_tree = traverse_util.unflatten_dict(delta_flat)

    result_vars = {
        key: value
        for key, value in base_unfrozen.items()
        if key not in ('params', 'base_params')
    }
    result_vars['base_params'] = base_tree
    result_vars['params'] = {'delta': delta_tree}
    return flax_core.freeze(result_vars)


@pytest.mark.skipif(torch.cuda.is_available(), reason='CPU-only reference test')
def test_finetune_gradient_parity(tmp_path, mace_model_path):
    pytest.importorskip('mace')
    pytest.importorskip('mace_jax')

    structures = _build_match_structures()
    train_subset = tmp_path / 'train_subset.h5'
    valid_subset = tmp_path / 'valid_subset.h5'
    _write_match_dataset(train_subset, structures)
    _write_match_dataset(valid_subset, structures)

    args_torch = _build_torch_args(
        train_subset,
        valid_subset,
        tmp_path / 'torch_grad',
        mace_model_path,
        max_steps=1,
        lr=1e-4,
    )
    args_torch.model = args_torch.model.float().train()

    torch_batch = _make_torch_batch(structures, args_torch.model)
    torch_energy = args_torch.model(torch_batch)['energy']
    num_atoms = torch_batch.ptr[1:] - torch_batch.ptr[:-1]
    energy_weights = torch.ones_like(torch_energy) / num_atoms.to(torch_energy.dtype)
    torch_loss = (F.huber_loss(
        torch_energy,
        torch.zeros_like(torch_energy),
        delta=0.01,
        reduction='none',
    ) * energy_weights).mean()
    args_torch.model.zero_grad(set_to_none=True)
    torch_loss.backward()
    torch_grad_vec = torch.cat([
        param.grad.reshape(-1)
        for param in args_torch.model.parameters()
    ]).detach().cpu().numpy()

    jax_model_dir = tmp_path / 'jax_grad'
    _export_jax_model(
        mace_model_path,
        [int(z) for z in args_torch.model.atomic_numbers],
        list(args_torch.model.atomic_energies),
        args_torch.model.r_max,
        jax_model_dir,
    )
    config_path = jax_model_dir / DEFAULT_CONFIG_NAME
    config_data = json.loads(config_path.read_text())
    config_data['train_deltas'] = True
    config_path.write_text(json.dumps(config_data))

    args_jax = _build_jax_args(
        train_subset,
        valid_subset,
        tmp_path / 'jax_grad',
        jax_model_dir,
        max_steps=1,
        lr=1e-4,
    )

    with _patch_jax_loader_for_deltas():
        bundle = load_model_bundle(str(jax_model_dir), dtype='float32')
        z_table = JaxAtomicNumberTable(tuple(bundle.config['atomic_numbers']))
        graphs = atoms_to_graphs(str(train_subset), bundle.config['r_max'], z_table)
        loader = build_loader(
            graphs,
            batch_size=1,
            shuffle=False,
            max_nodes=None,
            max_edges=None,
        )
        graph = next(iter(loader))

        wrapper = JaxMaceWrapper(
            module=bundle.module,
            config=bundle.config,
            compute_force=False,
            compute_stress=False,
        )
        apply_fn = make_apply_fn(wrapper, num_species=len(z_table))
        loss_settings = LossSettings.from_args(args_jax)
        loss_fn = build_loss_fn(apply_fn, loss_settings)

        def scalar_loss(variables):
            total_loss_value, _ = loss_fn(variables, graph)
            return total_loss_value

        jax_loss_value, _ = loss_fn(bundle.params, graph)
        jax_grads = jax.grad(scalar_loss)(bundle.params)
        flat_jax = traverse_util.flatten_dict(
            jtu.tree_map(lambda x: np.asarray(x), jax_grads['params']['delta']),
        )
        jax_grad_vec = np.concatenate([
            flat_jax[key].ravel()
            for key in sorted(flat_jax)
        ])

    np.testing.assert_allclose(
        float(jax_loss_value),
        float(torch_loss.detach().cpu().numpy()),
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.linalg.norm(jax_grad_vec),
        np.linalg.norm(torch_grad_vec),
        rtol=1e-4,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.max(np.abs(jax_grad_vec)),
        np.max(np.abs(torch_grad_vec)),
        rtol=1e-4,
        atol=1e-7,
    )


@pytest.mark.skipif(torch.cuda.is_available(), reason='CPU-only reference test')
def test_finetune_mace_jax(tmp_path, mace_model_path):
    pytest.importorskip('mace')
    pytest.importorskip('mace_jax')

    data_dir = Path(__file__).with_name('data')
    train_file = data_dir / 'train.h5'
    valid_file = data_dir / 'valid.h5'
    # Use synthetic parity dataset to ensure deterministic matching between frameworks.
    parity_structures = _build_match_structures()
    train_subset = tmp_path / 'train_subset.h5'
    valid_subset = tmp_path / 'valid_subset.h5'
    _write_match_dataset(train_subset, parity_structures)
    _write_match_dataset(valid_subset, parity_structures)
    structures = _load_structures(train_subset)

    args_torch = _build_torch_args(
        train_subset,
        valid_subset,
        tmp_path / 'torch_out',
        mace_model_path,
    )

    torch_model_pre = args_torch.model.float().eval()
    torch_energy_pre = _predict_torch_energy(torch_model_pre, structures)

    equitrain_train(args_torch)

    torch_model_post = args_torch.model.float().eval()
    torch_energy_post = _predict_torch_energy(torch_model_post, structures)

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

    with _patch_jax_loader_for_deltas():
        bundle = load_model_bundle(str(jax_model_dir), dtype='float32')
        data_dict = _structures_to_jax_input(structures, torch_model_pre)
        jax_energy_pre = _predict_jax_energy(bundle, data_dict)

        np.testing.assert_allclose(
            jax_energy_pre,
            torch_energy_pre,
            rtol=1e-5,
            atol=1e-4,
            err_msg='Torch and JAX predictions differ before fine-tuning.',
        )

        torch_post = torch.load(torch_post_path, weights_only=False).float().eval()
        jax_params_from_torch = _convert_torch_model_to_jax_params(
            torch_post,
            atomic_numbers,
            base_params=bundle.params,
        )
        jax_energy_post = _predict_jax_energy(
            bundle,
            data_dict,
            params=jax_params_from_torch,
        )
        np.testing.assert_allclose(
            jax_energy_post,
            torch_energy_post,
            rtol=1e-5,
            atol=1e-5,
            err_msg='Torch and JAX predictions differ after fine-tuning.',
        )

        delta_atomic = np.asarray(
            serialization.to_state_dict(jax_params_from_torch)['params']['delta'][
                'atomic_energies_fn'
            ]['atomic_energies']
        )
        np.testing.assert_allclose(
            delta_atomic,
            0.0,
            atol=0.0,
            err_msg='Delta parameters modified atomic energies.',
        )


@pytest.mark.skipif(torch.cuda.is_available(), reason='CPU-only reference test')
def test_jax_checkpoint_parity(tmp_path, mace_model_path):
    pytest.importorskip('mace')
    pytest.importorskip('mace_jax')

    data_dir = Path(__file__).with_name('data')
    train_file = data_dir / 'train.h5'
    valid_file = data_dir / 'valid.h5'
    parity_structures = _build_match_structures()
    train_subset = tmp_path / 'train_subset.h5'
    valid_subset = tmp_path / 'valid_subset.h5'
    _write_match_dataset(train_subset, parity_structures)
    _write_match_dataset(valid_subset, parity_structures)
    structures = _load_structures(train_subset)

    torch_base_model = torch.load(mace_model_path, weights_only=False).float().eval()
    torch_pre_path = tmp_path / 'torch_pre.model'
    torch.save(torch_base_model, torch_pre_path)
    atomic_numbers = [int(z) for z in torch_base_model.atomic_numbers]
    atomic_energies = (
        torch_base_model.atomic_energies_fn.atomic_energies.detach().cpu().tolist()
    )
    r_max = torch_base_model.r_max.item()

    args_torch = _build_torch_args(
        train_subset,
        valid_subset,
        tmp_path / 'torch_checkpoint',
        mace_model_path,
        max_steps=_PARITY_STEPS,
        lr=1e-4,
    )
    args_torch.model = args_torch.model.float()

    equitrain_train(args_torch)

    torch_predictions = _predict_torch_energy(
        args_torch.model.float().eval(),
        structures,
    )

    torch_best_dir = _find_best_checkpoint_dir(Path(args_torch.output_dir))
    torch_model_path = torch_best_dir / 'pytorch_model.bin'
    if not torch_model_path.exists():
        torch_model_path = torch_best_dir / 'model.safetensors'
    assert torch_model_path.exists(), 'Torch checkpoint missing model weights.'

    args_torch_reload = _build_torch_args(
        train_subset,
        valid_subset,
        tmp_path / 'torch_reload',
        mace_model_path,
        max_steps=_PARITY_STEPS,
        lr=1e-4,
    )
    reloaded_torch = args_torch_reload.model.float().eval()
    load_torch_model_state(reloaded_torch, str(torch_model_path))
    torch_predictions_reload = _predict_torch_energy(
        reloaded_torch.float().eval(),
        structures,
    )
    np.testing.assert_allclose(
        torch_predictions_reload,
        torch_predictions,
        rtol=1e-5,
        atol=1e-5,
        err_msg='Torch checkpoint reload altered predictions.',
    )
    torch_export_path = tmp_path / 'torch_finetuned.model'
    args_torch.model.export(str(torch_export_path))

    jax_model_dir = tmp_path / 'jax_model'
    _export_jax_model(
        torch_pre_path,
        atomic_numbers,
        atomic_energies,
        r_max,
        jax_model_dir,
    )
    config_path = jax_model_dir / DEFAULT_CONFIG_NAME
    config_data = json.loads(config_path.read_text())
    config_data['train_deltas'] = True
    config_path.write_text(json.dumps(config_data))

    args_jax = _build_jax_args(
        train_subset,
        valid_subset,
        tmp_path / 'jax_checkpoint',
        jax_model_dir,
        max_steps=_PARITY_STEPS,
        lr=1e-4,
    )

    with _patch_jax_loader_for_deltas():
        base_bundle = load_model_bundle(str(jax_model_dir), dtype='float32')
        equitrain_train(args_jax)

        jax_best_dir = _find_best_checkpoint_dir(Path(args_jax.output_dir))
        bundle = load_model_bundle(str(jax_best_dir), dtype='float32')
        data_dict = _structures_to_jax_input(structures, args_torch.model)
        torch_post = torch.load(torch_export_path, weights_only=False).float().eval()
        jax_params_from_torch = _convert_torch_model_to_jax_params(
            torch_post,
            atomic_numbers,
            base_params=bundle.params,
        )
        jax_predictions_trained = _predict_jax_energy(bundle, data_dict)
        jax_predictions_converted = _predict_jax_energy(
            bundle,
            data_dict,
            params=jax_params_from_torch,
        )

    np.testing.assert_allclose(
        jax_predictions_trained,
        torch_predictions,
        rtol=1e-5,
        atol=1e-5,
        err_msg='Checkpointed JAX model predictions differ from Torch.',
    )

    np.testing.assert_allclose(
        jax_predictions_converted,
        torch_predictions,
        rtol=1e-5,
        atol=1e-5,
        err_msg='Converted Torch parameters differ from Torch predictions.',
    )

    trained_delta = serialization.to_state_dict(bundle.params)['params']['delta']
    converted_delta = serialization.to_state_dict(jax_params_from_torch)['params']['delta']
    flat_trained = traverse_util.flatten_dict(trained_delta)
    flat_converted = traverse_util.flatten_dict(converted_delta)
    for key, trained_val in flat_trained.items():
        key_str = '.'.join(key)
        np.testing.assert_allclose(
        np.asarray(trained_val),
        np.asarray(flat_converted[key]),
        rtol=0.0,
        atol=1e-8,
        err_msg=f'Fine-tuned delta mismatch at {key_str}',
    )

    delta_state = serialization.to_state_dict(jax_params_from_torch)['params']['delta']
    np.testing.assert_allclose(
        np.asarray(delta_state['atomic_energies_fn']['atomic_energies']),
        0.0,
        atol=0.0,
        err_msg='Delta parameters modified atomic energies after checkpoint conversion.',
    )

    base_state = serialization.to_state_dict(base_bundle.params)['params']['base']
    ckpt_state = serialization.to_state_dict(bundle.params)['params']['base']
    flat_base = traverse_util.flatten_dict(flax_core.unfreeze(base_state))
    flat_ckpt = traverse_util.flatten_dict(flax_core.unfreeze(ckpt_state))
    for key, base_val in flat_base.items():
        key_str = '.'.join(key)
        np.testing.assert_allclose(
            np.asarray(flat_ckpt[key]),
            np.asarray(base_val),
            atol=0.0,
            err_msg=f'Base parameter changed during JAX fine-tuning at {key_str}',
        )
