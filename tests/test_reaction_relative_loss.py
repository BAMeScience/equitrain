from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from equitrain.argparser import (
    ArgumentError,
    get_args_parser_train,
    validate_training_args,
)
from equitrain.backends.torch_loss_fn import LossFnCollection
from equitrain.backends.torch_loss_metrics import LossMetrics
from equitrain.data.backend_torch.loaders_impl import DynamicGraphCollater
from equitrain.data.backend_torch.loaders_reaction import (
    ReactionGraphCollater,
    ReactionGroupBatchSampler,
    _reaction_group_ids,
    _validate_relative_reaction_metadata,
)


def _loss_args(**overrides):
    args = dict(
        energy_weight=0.0,
        forces_weight=0.0,
        stress_weight=0.0,
        barrier_weight=1.0,
        reaction_energy_weight=1.0,
        loss_energy_per_atom=True,
        loss_type='mae',
        loss_type_energy='mae',
        loss_type_forces='mae',
        loss_type_stress='mae',
        loss_weight_type=None,
        loss_weight_type_energy=None,
        loss_weight_type_forces=None,
        loss_weight_type_stress=None,
        smooth_l1_beta=1.0,
        huber_delta=1.0,
        loss_clipping=None,
        loss_monitor=['mse'],
    )
    args.update(overrides)
    return args


class _MetadataDataset(torch.utils.data.Dataset):
    def __init__(self, metadata):
        self.metadata = metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        raise IndexError(index)

    def reaction_metadata(self):
        return self.metadata


def _reaction_graph(*, energy, reaction_group_id, state_id, num_nodes=1):
    pytest.importorskip('torch_geometric')
    from torch_geometric.data import Data

    return Data(
        pos=torch.zeros(num_nodes, 3),
        positions=torch.zeros(num_nodes, 3),
        y=torch.tensor(float(energy)),
        force=torch.zeros(num_nodes, 3),
        stress=torch.zeros(1, 3, 3),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        reaction_group_id=torch.tensor(reaction_group_id, dtype=torch.long),
        reaction_id=torch.tensor(reaction_group_id, dtype=torch.long),
        state_id=torch.tensor(state_id, dtype=torch.long),
    )


def test_reaction_group_ids_are_global_across_concat_datasets():
    left = _MetadataDataset([(1, 42, 0)])
    right = _MetadataDataset([(1, 42, 1), (1, 42, 2), (2, 42, 0)])
    dataset = torch.utils.data.ConcatDataset([left, right])

    assert _reaction_group_ids(dataset) == [0, 0, 0, 1]


def test_relative_reaction_metadata_requires_complete_requested_roles():
    args = SimpleNamespace(barrier_weight=1.0, reaction_energy_weight=0.0)

    with pytest.raises(
        ValueError, match='No complete reaction groups.*--barrier-weight'
    ):
        _validate_relative_reaction_metadata(
            args,
            [(1, 7, 0), (1, 7, 2)],
            label='train.h5',
        )


def test_relative_reaction_metadata_warns_for_incomplete_requested_roles():
    args = SimpleNamespace(barrier_weight=1.0, reaction_energy_weight=0.0)

    with pytest.warns(RuntimeWarning, match='missing roles required'):
        _validate_relative_reaction_metadata(
            args,
            [(1, 7, 0), (1, 7, 1), (1, 8, 0)],
            label='train.h5',
        )


def test_relative_reaction_losses_are_averaged_per_reaction_not_frame():
    pytest.importorskip('torch_geometric')
    from torch_geometric.data import Batch

    target = Batch.from_data_list(
        [
            _reaction_graph(energy=0.0, reaction_group_id=0, state_id=0, num_nodes=3),
            _reaction_graph(energy=5.0, reaction_group_id=0, state_id=1, num_nodes=1),
            _reaction_graph(energy=-1.0, reaction_group_id=0, state_id=2, num_nodes=2),
            _reaction_graph(energy=100.0, reaction_group_id=-1, state_id=-1),
        ]
    )
    pred = {
        'energy': torch.tensor([0.0, 7.0, -0.5, 50.0]),
        'forces': torch.zeros(target.num_nodes, 3),
        'stress': torch.zeros(target.num_graphs, 3, 3),
    }

    loss, _ = LossFnCollection(
        **_loss_args(barrier_weight=2.0, reaction_energy_weight=3.0)
    )(pred, target)

    assert loss.main['barrier'].value.item() == pytest.approx(2.0)
    assert loss.main['barrier'].n.item() == pytest.approx(1.0)
    assert loss.main['reaction_energy'].value.item() == pytest.approx(0.5)
    assert loss.main['reaction_energy'].n.item() == pytest.approx(1.0)
    assert loss.main['total'].value.item() == pytest.approx(5.5)

    assert loss['mse']['barrier'].value.item() == pytest.approx(4.0)
    assert loss['mse']['reaction_energy'].value.item() == pytest.approx(0.25)
    assert loss['mse']['total'].value.item() == pytest.approx(8.75)


def test_relative_loss_metric_total_uses_reaction_averages():
    pytest.importorskip('torch_geometric')
    from torch_geometric.data import Batch

    args = SimpleNamespace(
        energy_weight=0.0,
        forces_weight=0.0,
        stress_weight=0.0,
        barrier_weight=1.0,
        reaction_energy_weight=0.0,
        loss_type='mae',
        loss_monitor=[],
    )
    loss_fn = LossFnCollection(
        **_loss_args(reaction_energy_weight=0.0, loss_monitor=[])
    )
    metrics = LossMetrics(args)

    target_1 = Batch.from_data_list(
        [
            _reaction_graph(energy=0.0, reaction_group_id=0, state_id=0),
            _reaction_graph(energy=0.0, reaction_group_id=0, state_id=1),
            *[
                _reaction_graph(energy=0.0, reaction_group_id=-1, state_id=-1)
                for _ in range(8)
            ],
        ]
    )
    loss_1, _ = loss_fn(
        {
            'energy': torch.tensor([0.0, 10.0] + [0.0] * 8),
            'forces': torch.zeros(target_1.num_nodes, 3),
            'stress': torch.zeros(target_1.num_graphs, 3, 3),
        },
        target_1,
    )
    metrics.update(loss_1)

    target_2 = Batch.from_data_list(
        [
            _reaction_graph(energy=0.0, reaction_group_id=1, state_id=0),
            _reaction_graph(energy=0.0, reaction_group_id=1, state_id=1),
        ]
    )
    loss_2, _ = loss_fn(
        {
            'energy': torch.tensor([0.0, 2.0]),
            'forces': torch.zeros(target_2.num_nodes, 3),
            'stress': torch.zeros(target_2.num_graphs, 3, 3),
        },
        target_2,
    )
    metrics.update(loss_2)

    assert metrics.main['barrier'].avg == pytest.approx(6.0)
    assert metrics.main['total'].avg == pytest.approx(6.0)


def test_relative_reaction_losses_ignore_batches_without_complete_roles():
    pytest.importorskip('torch_geometric')
    from torch_geometric.data import Batch

    target = Batch.from_data_list(
        [
            _reaction_graph(energy=0.0, reaction_group_id=0, state_id=0),
            _reaction_graph(energy=5.0, reaction_group_id=0, state_id=1),
        ]
    )
    pred = {
        'energy': torch.tensor([1.0, 5.0]),
        'forces': torch.zeros(target.num_nodes, 3),
        'stress': torch.zeros(target.num_graphs, 3, 3),
    }

    loss, _ = LossFnCollection(
        **_loss_args(barrier_weight=1.0, reaction_energy_weight=1.0)
    )(pred, target)

    assert loss.main['barrier'].n.item() == pytest.approx(1.0)
    assert loss.main['reaction_energy'].n.item() == pytest.approx(0.0)
    assert loss.main['total'].value.item() == pytest.approx(1.0)


def test_reaction_group_batch_sampler_keeps_groups_atomic():
    sampler = ReactionGroupBatchSampler(
        [0, -1, 0, 1, 1],
        batch_size=2,
        shuffle=False,
    )

    assert list(sampler) == [[0, 2], [1], [3, 4]]


def test_reaction_graph_collater_does_not_split_reaction_groups_across_sub_batches():
    items = [
        _reaction_graph(energy=0.0, reaction_group_id=0, state_id=0),
        _reaction_graph(energy=1.0, reaction_group_id=-1, state_id=-1, num_nodes=2),
        _reaction_graph(energy=2.0, reaction_group_id=0, state_id=1),
    ]
    collater = ReactionGraphCollater(
        lambda graphs: [int(graph.y.item()) for graph in graphs],
        max_nodes=2,
        max_edges=None,
        drop=False,
    )

    assert collater(items) == [[0, 2], [1]]


def test_dynamic_collater_keeps_standard_per_graph_batching():
    items = [
        _reaction_graph(energy=0.0, reaction_group_id=0, state_id=0),
        _reaction_graph(energy=1.0, reaction_group_id=-1, state_id=-1, num_nodes=2),
        _reaction_graph(energy=2.0, reaction_group_id=0, state_id=1),
    ]
    collater = DynamicGraphCollater(
        lambda graphs: [int(graph.y.item()) for graph in graphs],
        max_nodes=2,
        max_edges=None,
        drop=False,
    )

    assert collater(items) == [[0], [1], [2]]


def test_reaction_graph_collater_rejects_oversized_reaction_group():
    items = [
        _reaction_graph(energy=0.0, reaction_group_id=0, state_id=0, num_nodes=2),
        _reaction_graph(energy=2.0, reaction_group_id=0, state_id=1, num_nodes=2),
    ]
    collater = ReactionGraphCollater(
        lambda graphs: graphs,
        max_nodes=3,
        max_edges=None,
        drop=True,
    )

    with pytest.raises(ValueError, match='reaction groups cannot be split'):
        collater(items)


def test_reaction_group_batch_sampler_shards_evenly_by_whole_batches():
    rank_0 = ReactionGroupBatchSampler(
        [0, -1, 0, 1, 1],
        batch_size=2,
        shuffle=False,
        num_replicas=2,
        rank=0,
    )
    rank_1 = ReactionGroupBatchSampler(
        [0, -1, 0, 1, 1],
        batch_size=2,
        shuffle=False,
        num_replicas=2,
        rank=1,
    )

    assert not hasattr(rank_0, 'batch_size')
    assert len(rank_0) == len(rank_1) == 2
    assert list(rank_0) == [[0, 2], [3, 4]]
    assert list(rank_1) == [[1], [0, 2]]


def test_reaction_group_batch_sampler_shuffle_is_epoch_seeded_across_ranks():
    group_ids = [0, -1, 0, 1, 1, -1]
    global_sampler = ReactionGroupBatchSampler(
        group_ids,
        batch_size=2,
        shuffle=True,
        seed=17,
    )
    rank_0 = ReactionGroupBatchSampler(
        group_ids,
        batch_size=2,
        shuffle=True,
        seed=17,
        num_replicas=2,
        rank=0,
    )
    rank_1 = ReactionGroupBatchSampler(
        group_ids,
        batch_size=2,
        shuffle=True,
        seed=17,
        num_replicas=2,
        rank=1,
    )
    for sampler in (global_sampler, rank_0, rank_1):
        sampler.set_epoch(4)

    global_batches = list(global_sampler)
    if len(global_batches) % 2:
        global_batches.append(global_batches[0])
    interleaved = [
        batch for pair in zip(list(rank_0), list(rank_1), strict=True) for batch in pair
    ]

    assert interleaved == global_batches


def test_jax_rejects_relative_reaction_losses():
    args = get_args_parser_train().parse_args([])
    args.train_file = 'train.h5'
    args.valid_file = 'valid.h5'
    args.output_dir = 'out'
    args.model = 'model'
    args.energy_weight = 0.0
    args.forces_weight = 0.0
    args.stress_weight = 0.0
    args.barrier_weight = 1.0
    args.reaction_energy_weight = 0.0

    with pytest.raises(ArgumentError, match='JAX backend does not support'):
        validate_training_args(args, 'jax')
