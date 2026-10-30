from __future__ import annotations

import h5py
import pytest
import torch
from accelerate import Accelerator
from torch_geometric.data import Batch, Data

from equitrain.argparser import get_args_parser_train
from equitrain.backends.torch_evaluate import evaluate_main
from equitrain.backends.torch_loss_fn import LossFn
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_torch.equiload_loader import EquiLoadGraphLoader
from equitrain.data.backend_torch.loaders import get_dataloader
from equitrain.data.backend_torch.loaders_impl import DynamicGraphCollater
from equitrain.data.backend_torch.loaders_reaction import ReactionGraphCollater
from equitrain.data.format_hdf5 import HDF5Dataset
from equitrain.data.format_xyz.reader import XYZReader


def _dataset(tmp_path, name, energies):
    xyz = tmp_path / f'{name}.xyz'
    xyz.write_text(
        ''.join(
            '1\nLattice="5 0 0 0 5 0 0 0 5" Properties=species:S:1:pos:R:3 '
            f'REF_energy={energy} pbc="F F F"\nH 0 0 0\n'
            for energy in energies
        )
    )
    path = tmp_path / f'{name}.h5'
    reader = XYZReader(
        str(xyz),
        energy_key='REF_energy',
        forces_key='REF_forces',
        stress_key='REF_stress',
    )
    with HDF5Dataset(path, mode='w') as dataset:
        for index, atoms in enumerate(reader):
            dataset[index] = atoms
    return path


def _args():
    args = get_args_parser_train().parse_args([])
    args.loss_type = 'mae'
    args.loss_monitor = []
    args.energy_weight = 1.0
    args.forces_weight = args.stress_weight = 0.0
    args.loss_energy_per_atom = False
    args.batch_size = 1
    args.shuffle = False
    args.num_workers = 0
    return args


class _ZeroEnergy(torch.nn.Module):
    def forward(self, batch):
        return {
            'energy': torch.zeros_like(batch.y),
            'forces': torch.ones_like(batch.force),
            'stress': torch.ones_like(batch.stress),
        }


def test_multifile_errors_have_global_indices(tmp_path):
    paths = [_dataset(tmp_path, 'a', [1, 2]), _dataset(tmp_path, 'b', [3, 4])]
    args = _args()
    accelerator = Accelerator(cpu=True)
    loader = get_dataloader(args, paths, AtomicNumberTable([1]), 1.0, accelerator)
    assert [batch.idx.item() for group in loader for batch in group] == [0, 1, 2, 3]
    metrics, errors = evaluate_main(args, _ZeroEnergy(), accelerator, loader)
    torch.testing.assert_close(errors, errors.new_tensor([1, 2, 3, 4]))
    assert metrics.main['energy'].avg == pytest.approx(2.5)


def test_missing_xyz_labels_keep_zero_weights_through_hdf5_and_loss(tmp_path):
    path = _dataset(tmp_path, 'energy_only', [1])
    args = _args()
    args.forces_weight = args.stress_weight = 1.0
    loader = get_dataloader(args, path, AtomicNumberTable([1]), 1.0)
    batch = next(iter(loader))[0]
    assert batch.energy_weight.item() == 1
    assert batch.forces_weight.item() == batch.stress_weight.item() == 0
    loss, _ = LossFn(**vars(args))(_ZeroEnergy()(batch), batch)
    assert loss['energy'].value.item() == 1
    for key in ('forces', 'stress'):
        assert loss[key].value.item() == loss[key].n.item() == 0


@pytest.mark.parametrize('native_has_weights', [False, True])
def test_equiload_adapter_preserves_weights_across_shuffled_shards(
    tmp_path, native_has_weights
):
    paths = [_dataset(tmp_path, 'a', [1, 2]), _dataset(tmp_path, 'b', [3])]
    for path, energy_weight in zip(paths, [0.0, 2.0], strict=True):
        with h5py.File(path, 'r+') as handle:
            records = handle['structures'][:]
            records['energy_weight'][0] = energy_weight
            handle['structures'][:] = records
    args = _args()
    loader = get_dataloader(args, paths, AtomicNumberTable([1]), 1.0)
    batch = Batch.from_data_list([loader.dataset[index] for index in [2, 0, 2]])
    if not native_has_weights:
        for key in ('energy_weight', 'forces_weight', 'stress_weight'):
            del batch[key]

    class NativeLoader:
        dataset = loader.dataset

        def __len__(self):
            return 1

        def __iter__(self):
            yield [batch]

    adapter = EquiLoadGraphLoader(NativeLoader(), torch.device('cpu'), paths)
    result = next(iter(adapter))[0]
    assert result.idx.tolist() == [2, 0, 2]
    torch.testing.assert_close(result.energy_weight, result.y.new_tensor([2, 0, 2]))
    torch.testing.assert_close(result.forces_weight, torch.zeros_like(result.y))
    torch.testing.assert_close(result.stress_weight, torch.zeros_like(result.y))
    args.forces_weight = args.stress_weight = 1.0
    loss, _ = LossFn(**vars(args))(_ZeroEnergy()(result), result)
    assert loss['total'].value.item() == pytest.approx(6.0)


@pytest.mark.parametrize('component', ['energy', 'forces', 'stress'])
def test_zero_weight_nan_targets_have_zero_gradient(component):
    graphs = []
    for index in range(2):
        graphs.append(
            Data(
                pos=torch.zeros(1, 3),
                y=torch.tensor(float(index)),
                force=torch.full((1, 3), float(index)),
                stress=torch.full((1, 3, 3), float(index)),
                energy_weight=torch.tensor(float(index * 2)),
                forces_weight=torch.tensor(float(index * 2)),
                stress_weight=torch.tensor(float(index * 2)),
            )
        )
    target = Batch.from_data_list(graphs)
    true_key = {'energy': 'y', 'forces': 'force', 'stress': 'stress'}[component]
    target[true_key][0] = float('nan')
    outputs = {
        'energy': torch.zeros(2, requires_grad=True),
        'forces': torch.zeros(2, 3, requires_grad=True),
        'stress': torch.zeros(2, 3, 3, requires_grad=True),
    }
    args = _args()
    args.loss_type = 'mse'
    args.energy_weight = args.forces_weight = args.stress_weight = 0.0
    setattr(args, f'{component}_weight', 1.0)
    loss, errors = LossFn(**vars(args))(outputs, target)
    loss['total'].value.backward()
    assert loss[component].value.item() == pytest.approx(2)
    assert errors[0].item() == 0
    assert torch.all(outputs[component].grad[0] == 0)
    assert torch.isfinite(outputs[component].grad).all()


@pytest.mark.parametrize('collater_cls', [DynamicGraphCollater, ReactionGraphCollater])
def test_combined_limits_never_collate_an_empty_batch(collater_cls):
    small = Data(pos=torch.zeros(1, 3), edge_index=torch.empty(2, 0, dtype=torch.long))
    large = Data(pos=torch.zeros(2, 3), edge_index=torch.tensor([[0, 1], [1, 0]]))
    collater = collater_cls(Batch.from_data_list, max_nodes=2, max_edges=1)
    batches = collater([small, large])
    assert [batch.num_nodes for batch in batches] == [1, 2]
    assert [batch.num_graphs for batch in batches] == [1, 1]
