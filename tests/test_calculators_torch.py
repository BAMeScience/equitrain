from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('torch', reason='Torch is required for torch calculator tests.')
pytest.importorskip(
    'torch_geometric', reason='torch_geometric is required for torch calculator tests.'
)
pytest.importorskip('ase', reason='ASE is required for calculator tests.')

import torch  # noqa: E402
from ase import Atoms  # noqa: E402

import equitrain.calculators.torch_wrapper as torch_calculators  # noqa: E402
from equitrain.data.atomic import AtomicNumberTable  # noqa: E402


class _DummyTorchWrappedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.atomic_numbers = AtomicNumberTable([1, 8])
        self.r_max = 3.0

    def forward(self, batch):
        num_graphs = int(batch.ptr.shape[0] - 1)
        num_nodes = int(batch.ptr[-1].item())
        device = batch.ptr.device
        energy = torch.arange(1, num_graphs + 1, dtype=torch.float32, device=device)
        forces = torch.zeros((num_nodes, 3), dtype=torch.float32, device=device)
        return {'energy': energy, 'forces': forces}


def _install_dummy_builder(monkeypatch):
    dummy_model = _DummyTorchWrappedModel()

    def _fake_builder(**_kwargs):
        return dummy_model, 'cpu', 'mace'

    monkeypatch.setattr(
        torch_calculators,
        '_build_wrapped_torch_model',
        _fake_builder,
    )


def test_torch_wrapper_predictor_predict(monkeypatch):
    _install_dummy_builder(monkeypatch)
    predictor = torch_calculators.TorchWrapperPredictor(
        model='unused.pt',
        model_wrapper='mace',
        device='cpu',
        batch_size=8,
        require_forces=True,
    )
    atoms = [
        Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]]),
        Atoms(numbers=[8, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    ]
    energies, forces = predictor.predict(atoms, require_forces=True)

    np.testing.assert_allclose(np.asarray(energies), np.array([1.0, 2.0], dtype=float))
    assert forces is not None
    assert len(forces) == 2
    assert forces[0].shape == (2, 3)
    assert forces[1].shape == (2, 3)


def test_build_torch_ase_calculator(monkeypatch):
    _install_dummy_builder(monkeypatch)
    calc = torch_calculators.build_ase_calculator(
        model='unused.pt',
        model_wrapper='mace',
        device='cpu',
        batch_size=4,
    )
    atoms = Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]])
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert isinstance(float(energy), float)
    assert forces.shape == (2, 3)
