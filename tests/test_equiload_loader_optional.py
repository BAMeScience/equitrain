"""Optional-loader behavior must be tested even without the native extension."""

import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest
import torch

from equitrain.argparser import get_args_parser_train
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_torch.equiload_loader import _supported_torch_version
from equitrain.data.backend_torch.loaders import get_dataloader


@pytest.fixture
def loader_input():
    args = get_args_parser_train().parse_args([])
    args.num_workers = 0
    args.pin_memory = False
    args.shuffle = False
    args.batch_size = 2
    path = Path(__file__).with_name('data') / 'train.h5'
    with h5py.File(path) as handle:
        numbers = AtomicNumberTable(sorted(set(map(int, handle['atomic_numbers'][:]))))
    return args, path, numbers


def test_default_python_loader_works_without_equiload(monkeypatch, loader_input):
    monkeypatch.setitem(sys.modules, 'equiload', None)
    args, path, numbers = loader_input
    assert args.torch_loader == 'python'
    loader = get_dataloader(args, path, numbers, 3.0)
    assert next(iter(loader))[0].num_graphs == 2
    args.torch_loader = 'equiload'
    # Select a supported version so this checks the missing extension message.
    monkeypatch.setattr(torch, '__version__', '2.12.0')
    with pytest.raises(RuntimeError, match='separately installed EquiLoad'):
        get_dataloader(args, path, numbers, 3.0)


@pytest.mark.parametrize(
    'option',
    ['distributed', 'weighted_sampler', 'barrier_weight', 'reaction_energy_weight'],
)
def test_equiload_rejects_unsupported_modes(loader_input, option):
    args, path, numbers = loader_input
    args.torch_loader = 'equiload'
    accelerator = None
    if option == 'distributed':
        accelerator = SimpleNamespace(num_processes=2, device=torch.device('cpu'))
        message = 'single-process'
    else:
        setattr(args, option, 1)
        message = (
            'weighted resampling'
            if option == 'weighted_sampler'
            else 'relative reaction losses'
        )
    with pytest.raises(RuntimeError, match=message):
        get_dataloader(args, path, numbers, 3.0, accelerator)


@pytest.mark.parametrize(
    'version, supported',
    [('2.5.1', False), ('2.6.0', True), ('2.14.0+cu130', True), ('2.15.0', False)],
)
def test_equiload_torch_version_requirement(
    monkeypatch, loader_input, version, supported
):
    monkeypatch.setattr(torch, '__version__', version)
    assert _supported_torch_version() == supported
    if not supported:
        args, path, numbers = loader_input
        args.torch_loader = 'equiload'
        with pytest.raises(RuntimeError, match='supports Torch'):
            get_dataloader(args, path, numbers, 3.0)
