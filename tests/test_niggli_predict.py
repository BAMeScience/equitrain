from ase import Atoms
import torch
from torch_geometric.data import Data

from equitrain.data.atomic import AtomicNumberTable
from equitrain.backends import torch_predict as torch_predict_mod


def test_predict_atoms_applies_niggli_flag(monkeypatch):
    converted_atoms = []

    class StubAtomsToGraphs:
        def __init__(self, *args, **kwargs):
            pass

        def convert(self, atoms):
            converted_atoms.append(atoms)
            return Data(num_nodes=len(atoms))

    monkeypatch.setattr(
        'equitrain.backends.torch_predict.AtomsToGraphs', StubAtomsToGraphs
    )

    reduced_markers = []

    def fake_niggli_reduce(atoms):
        atoms.info['reduced'] = True
        reduced_markers.append(True)
        return atoms

    monkeypatch.setattr(
        'equitrain.backends.torch_predict.niggli_reduce_inplace', fake_niggli_reduce
    )

    class DummyModel(torch.nn.Module):
        def forward(self, batch):
            num_graphs = getattr(batch, 'num_graphs', 1)
            return {
                'energy': torch.zeros(num_graphs),
                'forces': None,
                'stress': None,
            }

    model = DummyModel()
    atoms = Atoms('Si2', positions=[[0, 0, 0], [0.5, 0.5, 0.5]])
    atoms.cell = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    atoms.pbc = [True, True, True]
    z_table = AtomicNumberTable([14])

    converted_atoms.clear()
    reduced_markers.clear()
    torch_predict_mod.predict_atoms(
        model,
        [atoms],
        z_table,
        r_max=5.0,
        batch_size=1,
        niggli_reduce=False,
    )
    assert not reduced_markers
    assert 'reduced' not in converted_atoms[-1].info

    converted_atoms.clear()
    reduced_markers.clear()
    torch_predict_mod.predict_atoms(
        model,
        [atoms],
        z_table,
        r_max=5.0,
        batch_size=1,
        niggli_reduce=True,
    )
    assert reduced_markers
    assert converted_atoms[-1].info.get('reduced') is True
