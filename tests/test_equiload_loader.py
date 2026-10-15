from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest
import torch

pytest.importorskip('equiload', reason='EquiLoad is an optional native extension.')

from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_torch.equiload_loader import _supported_torch_version
from equitrain.data.backend_torch.loaders import get_dataloader


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


def test_equiload_torch_version_requirement(monkeypatch):
    monkeypatch.setattr(torch, '__version__', '2.5.1')
    assert not _supported_torch_version()

    monkeypatch.setattr(torch, '__version__', '2.14.0+cu130')
    assert _supported_torch_version()


def test_equiload_loader_rejects_weighted_resampling():
    path = Path(__file__).with_name('data') / 'train.h5'
    args = SimpleNamespace(
        torch_loader='equiload',
        batch_size=2,
        shuffle=False,
        batch_max_nodes=None,
        batch_max_edges=None,
        batch_drop=False,
        seed=123,
        weighted_sampler=True,
        loss_type='MSE',
        relative_reaction_loss=False,
    )

    with pytest.raises(RuntimeError, match='weighted resampling'):
        get_dataloader(args, path, [1], r_max=5.0)
