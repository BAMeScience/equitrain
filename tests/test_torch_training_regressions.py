from __future__ import annotations

import os
import time
from datetime import timedelta

import pytest
import torch
from accelerate import Accelerator
from torch_ema import ExponentialMovingAverage
from torch_geometric.data import Batch, Data

from equitrain.argparser import get_args_parser_train
from equitrain.backends.torch_backend import train_one_epoch
from equitrain.backends.torch_evaluate import evaluate_main
from equitrain.backends.torch_loss_fn import LossFn
from equitrain.data.backend_torch.loaders_impl import DynamicGraphLoader
from equitrain.data.backend_torch.loaders_reaction import ReactionGraphLoader
from equitrain.logger import FileLogger


def _args(**overrides):
    args = get_args_parser_train().parse_args([])
    args.loss_type = 'mse'
    args.loss_monitor = []
    args.energy_weight = 1.0
    args.forces_weight = args.stress_weight = 0.0
    args.loss_energy_per_atom = False
    args.gradient_clipping = None
    args.verbose = 0
    args.tqdm = False
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _graph(energy, index, *, nodes=1, feature=1.0, group=-1, state=-1):
    return Data(
        pos=torch.zeros(nodes, 3),
        y=torch.tensor(float(energy)),
        force=torch.full((nodes, 3), float(energy + 1)),
        stress=torch.full((1, 3, 3), float(energy + 2)),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        idx=index,
        feature=torch.tensor(feature),
        reaction_group_id=torch.tensor(group),
        state_id=torch.tensor(state),
        energy_weight=torch.tensor(1.0),
        forces_weight=torch.tensor(1.0),
        stress_weight=torch.tensor(1.0),
    )


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(3))
        self.register_buffer('scale', torch.ones(()))
        self.calls = []

    def forward(self, data):
        self.calls.append(self.weight.detach().clone())
        return {
            'energy': self.weight[0] * data.feature * self.scale,
            'forces': self.weight[1].expand(data.num_nodes, 3),
            'stress': self.weight[2].expand(data.num_graphs, 3, 3),
        }


def _logger():
    return FileLogger(log_to_file=False, enable_logging=False, verbosity=0)


def _train(args, model, loader, accelerator, optimizer=None, ema=None):
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return train_one_epoch(
        args, model, ema, accelerator, loader, optimizer, None, 1, _logger()
    )


@pytest.mark.parametrize('node_limit', [None, 2, 3])
def test_energy_update_is_independent_of_sub_batch_sizes(node_limit):
    accelerator = Accelerator(cpu=True)
    graphs = [_graph(value, i) for i, value in enumerate([0, 0, 0, 4])]
    loader = accelerator.prepare(
        DynamicGraphLoader(graphs, batch_size=4, max_nodes=node_limit)
    )
    model = _Model()
    _train(_args(), model, loader, accelerator)
    assert model.weight[0].item() == pytest.approx(0.2)


def test_each_component_uses_its_own_labeled_element_count():
    args = _args(forces_weight=2.0, stress_weight=0.5)
    graphs = [_graph(i + 1, i, nodes=n) for i, n in enumerate([1, 3, 2, 4])]
    graphs[0].energy_weight.fill_(0)
    graphs[1].forces_weight.fill_(0)
    graphs[2].stress_weight.fill_(0)
    graphs[3].energy_weight.fill_(2)
    whole = Batch.from_data_list(graphs)
    reference = _Model()
    loss, _ = LossFn(**vars(args))(reference(whole), whole)
    loss['total'].value.backward()
    expected = -0.1 * reference.weight.grad

    accelerator = Accelerator(cpu=True)
    loader = accelerator.prepare(DynamicGraphLoader(graphs, batch_size=4, max_nodes=4))
    model = _Model()
    metrics, _ = _train(args, model, loader, accelerator)
    torch.testing.assert_close(model.weight, expected)
    assert metrics.main['total'].avg == pytest.approx(loss['total'].value.item())


def test_relative_losses_normalize_by_reactions_across_sub_batches():
    args = _args(energy_weight=0.25, barrier_weight=2.0, reaction_energy_weight=3.0)
    graphs = [
        _graph(y, i, feature=float(i % 3 + 1), group=i // 3, state=i % 3)
        for i, y in enumerate([0, 2, -1, 0, 5, -3])
    ]
    graphs.append(_graph(10, 6))
    batch = Batch.from_data_list(graphs)
    reference = _Model()
    loss, _ = LossFn(**vars(args))(reference(batch), batch)
    loss['total'].value.backward()
    expected = -0.1 * reference.weight.grad

    accelerator = Accelerator(cpu=True)
    loader = accelerator.prepare(ReactionGraphLoader(graphs, batch_size=7, max_nodes=3))
    model = _Model()
    _train(args, model, loader, accelerator)
    torch.testing.assert_close(model.weight, expected)


@pytest.mark.parametrize('limit, expected', [(None, 3), (0, 0), (1, 1), (2, 2), (9, 3)])
def test_training_and_evaluation_respect_step_counts(limit, expected):
    accelerator = Accelerator(cpu=True)
    loader = accelerator.prepare(
        DynamicGraphLoader([_graph(1, i) for i in range(3)], batch_size=1)
    )
    args = _args(train_max_steps=limit)
    model = _Model()
    _train(args, model, loader, accelerator)
    assert len(model.calls) == expected
    model.calls.clear()
    evaluate_main(args, model, accelerator, loader, max_steps=limit)
    assert len(model.calls) == expected


def test_ema_validation_restores_raw_training_weights_between_epochs():
    accelerator = Accelerator(cpu=True)
    loader = accelerator.prepare(DynamicGraphLoader([_graph(1, 0)], batch_size=1))
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9)
    args = _args()
    _train(args, model, loader, accelerator, optimizer, ema)
    assert model.weight[0].item() == pytest.approx(0.2)
    averaged = ema.shadow_params[0][0].item()
    assert averaged != pytest.approx(0.2)
    metrics, _ = evaluate_main(args, model, accelerator, loader, model_ema=ema)
    assert metrics.main['energy'].avg == pytest.approx((1 - averaged) ** 2)
    assert model.weight[0].item() == pytest.approx(0.2)
    _train(args, model, loader, accelerator, optimizer, ema)
    assert model.calls[-1][0].item() == pytest.approx(0.2)
    assert model.weight[0].item() == pytest.approx(0.36)


def test_ema_weights_are_restored_when_evaluation_raises():
    accelerator = Accelerator(cpu=True)
    model = _Model()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9)
    with torch.no_grad():
        model.weight.fill_(2)
    graph = _graph(1, 0)
    del graph.feature
    loader = accelerator.prepare(DynamicGraphLoader([graph], batch_size=1))
    with pytest.raises(AttributeError):
        evaluate_main(_args(), model, accelerator, loader, model_ema=ema)
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 2))


def test_training_checkpoints_keep_raw_weights_while_final_model_uses_ema(
    tmp_path, monkeypatch
):
    from equitrain.backends import torch_backend
    from equitrain.backends.torch_checkpoint import load_model_state

    model = _Model()
    model.atomic_numbers = [1]
    model.r_max = 1.0
    accelerator = Accelerator(cpu=True)
    loader = accelerator.prepare(DynamicGraphLoader([_graph(1, 0)], batch_size=1))
    args = _args(
        model=model,
        epochs=2,
        output_dir=str(tmp_path),
        ema=True,
        ema_decay=0.9,
        scheduler='step',
        gamma=1.0,
        step_size=1,
        weight_decay=0.0,
    )
    monkeypatch.setattr(torch_backend, 'get_model', lambda *a, **kw: model)
    monkeypatch.setattr(
        torch_backend, 'get_dataloaders', lambda *a: (loader, loader, None)
    )
    monkeypatch.setattr(
        torch_backend,
        'create_optimizer',
        lambda args, model: torch.optim.SGD(model.parameters(), lr=0.1),
    )
    torch_backend._train_with_accelerator(args, accelerator)
    expected_ema = 0.25 * (0.2 * 9 / 11) + 0.75 * 0.36
    assert model.weight[0].item() == pytest.approx(expected_ema)
    checkpoint = next(tmp_path.glob('best_val_epochs@2_*'))
    raw_model = _Model()
    load_model_state(raw_model, checkpoint / 'model.safetensors')
    assert raw_model.weight[0].item() == pytest.approx(0.36)


def test_all_dropped_batches_are_skipped_without_reusing_previous_graphs():
    accelerator = Accelerator(cpu=True)
    graphs = [_graph(1, 0, nodes=2), _graph(1, 1), _graph(1, 2, nodes=2)]
    loader = accelerator.prepare(
        DynamicGraphLoader(graphs, batch_size=1, max_nodes=1, drop=True)
    )
    model = _Model()
    _train(_args(batch_drop=True), model, loader, accelerator)
    assert len(model.calls) == 1
    assert model.weight[0].item() == pytest.approx(0.2)


def _distributed_worker(rank, rendezvous):
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE='2',
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE='2',
        MASTER_ADDR='127.0.0.1',
        MASTER_PORT='29864',
        OMP_NUM_THREADS='1',
    )
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        'gloo',
        init_method=f'file://{rendezvous}',
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        accelerator = Accelerator(cpu=True)
        # Three graphs require Accelerate to pad the last distributed batch.
        loader = accelerator.prepare(
            DynamicGraphLoader([_graph(i + 1, i) for i in range(3)], batch_size=1)
        )
        model = _Model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        model, optimizer = accelerator.prepare(model, optimizer)
        args = _args()
        _, errors = evaluate_main(args, model, accelerator, loader)
        torch.testing.assert_close(errors, errors.new_tensor([1, 4, 9]))
        _, errors = _train(args, model, loader, accelerator, optimizer)
        torch.testing.assert_close(errors, errors.new_tensor([1, 4, 9]))

        # Only rank zero loses its first batch; both must skip before DDP sync.
        loader = accelerator.prepare(
            DynamicGraphLoader(
                [_graph(1, 0, nodes=2), *[_graph(1, i) for i in range(1, 4)]],
                batch_size=1,
                max_nodes=1,
                drop=True,
            )
        )
        optimizer.param_groups[0]['lr'] = 0.1
        _train(_args(batch_drop=True), model, loader, accelerator, optimizer)
        assert model.module.weight[0].item() == pytest.approx(0.2)
        _, errors = evaluate_main(args, model, accelerator, loader)
        torch.testing.assert_close(errors, errors.new_tensor([0, 0.64, 0.64, 0.64]))

        # Unequal local graph counts still give the global mean gradient.
        with torch.no_grad():
            model.module.weight.zero_()
        graphs = [_graph(value, i) for i, value in enumerate([0, 0, 0, 4, 4, 4])]
        graphs[3].pos = torch.zeros(2, 3)
        graphs[3].force = torch.zeros(2, 3)
        loader = accelerator.prepare(
            DynamicGraphLoader(
                graphs,
                batch_size=3,
                max_nodes=1,
                drop=True,
            )
        )
        _train(_args(batch_drop=True), model, loader, accelerator, optimizer)
        assert model.module.weight[0].item() == pytest.approx(0.32)
    finally:
        torch.distributed.destroy_process_group()


def test_distributed_errors_padding_empty_batches_and_gradient_counts(tmp_path):
    context = torch.multiprocessing.spawn(
        _distributed_worker, args=(str(tmp_path / 'rendezvous'),), nprocs=2, join=False
    )
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail('Distributed training did not finish within 60 seconds.')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
