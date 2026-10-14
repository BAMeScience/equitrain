from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest

pytest.importorskip('rtorch', reason='RTorch is an optional native extension.')

from equitrain.data.backend_torch.loaders import get_dataloader


def test_optional_rtorch_loader_matches_equitrain_batch_contract():
    path = Path(__file__).with_name('data') / 'train.h5'
    with h5py.File(path) as handle:
        atomic_numbers = sorted({int(value) for value in handle['atomic_numbers'][:]})
    args = SimpleNamespace(
        torch_loader='rtorch',
        batch_size=2,
        shuffle=False,
        batch_max_nodes=None,
        batch_max_edges=None,
        batch_drop=False,
        seed=123,
        weighted_sampler=False,
        loss_type='MSE',
        relative_reaction_loss=False,
    )

    loader = get_dataloader(args, path, atomic_numbers, r_max=5.0)
    batches = next(iter(loader))

    assert loader._uses_rtorch
    assert len(batches) == 1
    assert batches[0].num_graphs == 2
    assert batches[0].pos.device.type == 'cpu'


def test_rtorch_loader_rejects_weighted_resampling():
    path = Path(__file__).with_name('data') / 'train.h5'
    args = SimpleNamespace(
        torch_loader='rtorch',
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
