from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from torch_geometric.data import Batch

pytest.importorskip('equiload', reason='EquiLoad is an optional native extension.')

from equitrain.backends.torch_loss_fn import LossFn
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_torch.loaders import get_dataloader
from equitrain.data.format_hdf5 import HDF5Dataset


def test_optional_equiload_loader_matches_equitrain_batch_contract():
    path = Path(__file__).with_name('data') / 'train.h5'
    with h5py.File(path) as handle:
        atomic_numbers = AtomicNumberTable(
            sorted({int(value) for value in handle['atomic_numbers'][:]})
        )
    args = SimpleNamespace(
        torch_loader='equiload',
        batch_size=2,
        shuffle=False,
        batch_max_nodes=None,
        batch_max_edges=None,
        batch_drop=False,
        seed=123,
        weighted_sampler=False,
        loss_type='MSE',
        relative_reaction_loss=False,
        pin_memory=False,
        num_workers=0,
    )

    loader = get_dataloader(args, path, atomic_numbers, r_max=5.0)
    batches = next(iter(loader))

    assert loader._uses_equiload
    assert len(batches) == 1
    assert batches[0].num_graphs == 2
    assert batches[0].pos.device.type == 'cpu'


def test_equiload_float_dtypes_match_python_loader():
    path = Path(__file__).with_name('data') / 'train.h5'
    with h5py.File(path) as handle:
        atomic_numbers = AtomicNumberTable(
            sorted({int(value) for value in handle['atomic_numbers'][:]})
        )
    common = dict(
        batch_size=2,
        shuffle=False,
        batch_max_nodes=None,
        batch_max_edges=None,
        batch_drop=False,
        seed=123,
        weighted_sampler=False,
        loss_type='MSE',
        relative_reaction_loss=False,
        pin_memory=False,
        num_workers=0,
    )
    python_batch = next(
        iter(
            get_dataloader(
                SimpleNamespace(torch_loader='python', **common),
                path,
                atomic_numbers,
                5.0,
            )
        )
    )[0]
    equiload_batch = next(
        iter(
            get_dataloader(
                SimpleNamespace(torch_loader='equiload', **common),
                path,
                atomic_numbers,
                5.0,
            )
        )
    )[0]

    assert set(equiload_batch.keys()) == set(python_batch.keys())
    for key in python_batch.keys():
        python_value = python_batch[key]
        equiload_value = equiload_batch[key]
        if torch.is_tensor(python_value):
            assert torch.is_tensor(equiload_value)
            assert equiload_value.dtype == python_value.dtype, key
    assert equiload_batch.cell_volume.dtype is torch.float64


@pytest.fixture(params=[1, 2], ids=['single_file', 'multiple_files'])
def weighted_shards(tmp_path, request):
    weights = [(1, 0, 0), (0, 2, 0.5), (2, 0.5, 0), (0, 0, 3), (0.25, 1, 2)]
    shards = [range(5)] if request.param == 1 else [range(2), range(2, 5)]
    paths = [tmp_path / f'shard_{index}.h5' for index in range(len(shards))]
    for path, indices in zip(paths, shards, strict=True):
        with HDF5Dataset(path, mode='w') as dataset:
            for row, index in enumerate(indices):
                nodes = 1 + index % 2
                atoms = Atoms(
                    numbers=[1] * nodes,
                    positions=[[0.5 * atom, 0, 0] for atom in range(nodes)],
                    cell=[5, 5, 5],
                    pbc=False,
                )
                energy_weight, forces_weight, stress_weight = weights[index]
                atoms.info.update(
                    energy_weight=energy_weight,
                    forces_weight=forces_weight,
                    stress_weight=stress_weight,
                )
                # Missing labels deliberately contain NaNs: their masks must
                # survive loading and suppress both loss and gradient.
                atoms.calc = SinglePointCalculator(
                    atoms,
                    energy=float(index + 1) if energy_weight else float('nan'),
                    forces=np.full((nodes, 3), index + 2 if forces_weight else np.nan),
                    stress=np.full(6, index + 3 if stress_weight else np.nan),
                )
                dataset[row] = atoms
    return paths, weights


def _loss_and_gradients(batch):
    outputs = {
        'energy': torch.full_like(batch.y, 0.25, requires_grad=True),
        'forces': torch.full_like(batch.force, 0.25, requires_grad=True),
        'stress': torch.full_like(batch.stress, 0.25, requires_grad=True),
    }
    loss, errors = LossFn(loss_type='mse', loss_energy_per_atom=False)(outputs, batch)
    gradients = torch.autograd.grad(loss['total'].value, tuple(outputs.values()))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
    assert torch.isfinite(errors).all()
    assert (gradients[0][batch.energy_weight == 0] == 0).all()
    assert (gradients[1][batch.forces_weight[batch.batch] == 0] == 0).all()
    assert (gradients[2][batch.stress_weight == 0] == 0).all()
    return loss, errors, gradients


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('max_nodes, drop', [(None, False), (3, False), (1, True)])
def test_native_equiload_weights_and_gradients_match_python(
    weighted_shards, dtype, shuffle, max_nodes, drop
):
    paths, weights = weighted_shards
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        common = dict(
            batch_size=4,
            shuffle=shuffle,
            batch_max_nodes=max_nodes,
            batch_max_edges=None,
            batch_drop=drop,
            seed=123,
            weighted_sampler=False,
            relative_reaction_loss=False,
            pin_memory=False,
            num_workers=0,
        )
        atomic_numbers = AtomicNumberTable([1])
        python_loader = get_dataloader(
            SimpleNamespace(torch_loader='python', **common),
            paths,
            atomic_numbers,
            1.0,
        )
        native_loader = get_dataloader(
            SimpleNamespace(torch_loader='equiload', **common),
            paths,
            atomic_numbers,
            1.0,
        )
        expected_weights = torch.tensor(weights, dtype=dtype)
        expected_indices = [0, 2, 4] if drop else list(range(5))
        for _ in range(2):
            seen = []
            for batches in native_loader:
                for batch in batches:
                    indices = batch.idx.tolist()
                    seen.extend(indices)
                    reference = Batch.from_data_list(
                        [python_loader.dataset[index] for index in indices]
                    )
                    for column, component in enumerate(('energy', 'forces', 'stress')):
                        actual = batch[f'{component}_weight']
                        assert actual.dtype == dtype
                        assert actual.device == batch.y.device
                        torch.testing.assert_close(
                            actual, expected_weights[batch.idx, column]
                        )
                    native_loss, native_errors, native_grad = _loss_and_gradients(batch)
                    python_loss, python_errors, python_grad = _loss_and_gradients(
                        reference
                    )
                    for component in ('energy', 'forces', 'stress', 'total'):
                        torch.testing.assert_close(
                            native_loss[component].value, python_loss[component].value
                        )
                        torch.testing.assert_close(
                            native_loss[component].n, python_loss[component].n
                        )
                    torch.testing.assert_close(native_errors, python_errors)
                    torch.testing.assert_close(native_grad, python_grad)
            assert sorted(seen) == expected_indices
            if shuffle:
                assert seen != expected_indices
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_equiload_mace_predictions_match_python_loader(dtype):
    pytest.importorskip('mace')
    from equitrain.argparser import get_args_parser_train
    from equitrain.utility_test import MaceWrapper
    from equitrain.utility_test.mace_support import get_mace_model_path

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        args = get_args_parser_train().parse_args([])
        args.shuffle = False
        args.batch_size = 2
        args.num_workers = 0
        args.pin_memory = False
        model = MaceWrapper(args, filename_model=get_mace_model_path()).to(dtype=dtype)
        model.eval()
        path = Path(__file__).with_name('data') / 'train.h5'
        predictions = {}
        for backend in ('python', 'equiload'):
            args.torch_loader = backend
            loader = get_dataloader(args, path, model.atomic_numbers, model.r_max)
            values = model(next(iter(loader))[0])
            predictions[backend] = {
                key: values[key].detach() for key in ('energy', 'forces', 'stress')
            }
        for key in ('energy', 'forces', 'stress'):
            torch.testing.assert_close(
                predictions['equiload'][key], predictions['python'][key]
            )
    finally:
        torch.set_default_dtype(previous_dtype)
