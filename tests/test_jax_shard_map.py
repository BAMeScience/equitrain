from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip(
    'jax', reason='JAX runtime is required for JAX backend tests.'
)
jraph = pytest.importorskip('jraph', reason='jraph is required for JAX backend tests.')
optax = pytest.importorskip('optax', reason='optax is required for JAX backend tests.')

import jax.numpy as jnp

from equitrain import get_args_parser_train
from equitrain.backends import jax_backend, jax_predict, jax_utils
from equitrain.backends.jax_backend import (
    TrainState,
    _build_eval_step,
    _build_train_functions,
    _multi_device_chunk_iterator,
)
from equitrain.backends.jax_predict import predict_streaming
from equitrain.backends.jax_utils import (
    prepare_sharded_batch,
    process_local_sharded_to_global,
    shard_graphs_for_devices,
)


def _minimal_graph(graph_ids):
    graph_ids = jnp.asarray(graph_ids, dtype=jnp.int32)
    graph_count = int(graph_ids.shape[0])
    real_graphs = max(graph_count - 1, 0)
    return jraph.GraphsTuple(
        nodes={'positions': jnp.zeros((real_graphs, 3), dtype=jnp.float32)},
        edges={'shifts': jnp.zeros((0, 3), dtype=jnp.float32)},
        senders=jnp.zeros((0,), dtype=jnp.int32),
        receivers=jnp.zeros((0,), dtype=jnp.int32),
        globals={
            'cell': jnp.zeros((graph_count, 3, 3), dtype=jnp.float32),
            'graph_id': graph_ids,
        },
        n_node=jnp.concatenate(
            [
                jnp.ones((real_graphs,), dtype=jnp.int32),
                jnp.zeros((1,), dtype=jnp.int32),
            ]
        ),
        n_edge=jnp.zeros((graph_count,), dtype=jnp.int32),
    )


def test_is_multi_device_uses_global_device_count(monkeypatch):
    monkeypatch.setattr(jax_utils.jax, 'devices', lambda: [object(), object()])

    assert jax_utils.is_multi_device() is True


def test_process_local_sharded_to_global_uses_global_device_axis():
    local_device_count = jax.local_device_count()
    global_device_count = len(jax.devices())
    local_batch = {
        'x': jnp.arange(local_device_count * 2, dtype=jnp.float32).reshape(
            local_device_count, 2
        )
    }

    global_batch = process_local_sharded_to_global(local_batch)

    assert global_batch['x'].shape == (global_device_count, 2)
    assert global_batch['x'].sharding.spec == jax.sharding.PartitionSpec('device')


def test_distributed_chunk_iterator_pads_exhausted_process(monkeypatch):
    graph = _minimal_graph([0, -1])
    gathered_states = [
        np.array([1, 1], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
        np.array([0, 0], dtype=np.int32),
    ]

    from jax.experimental import multihost_utils

    def fake_process_allgather(_value, *, tiled=False):
        assert tiled is False
        return gathered_states.pop(0)

    class Logger:
        def __init__(self):
            self.messages = []

        def log(self, _level, message):
            self.messages.append(message)

    monkeypatch.setattr(multihost_utils, 'process_allgather', fake_process_allgather)
    logger = Logger()

    chunks = list(
        _multi_device_chunk_iterator(
            [graph],
            2,
            phase='Training',
            logger=logger,
            sync_processes=True,
        )
    )

    assert len(chunks) == 2
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 2
    assert int(np.asarray(chunks[0][0].n_node).sum()) == 1
    assert int(np.asarray(chunks[0][1].n_node).sum()) == 0
    assert int(np.asarray(chunks[1][0].n_node).sum()) == 0
    assert int(np.asarray(chunks[1][1].n_node).sum()) == 0
    assert not gathered_states
    assert logger.messages


def test_train_final_test_eval_runs_on_non_primary_process(monkeypatch, tmp_path):
    args = get_args_parser_train().parse_args([])
    args.backend = 'jax'
    args.model = 'dummy-model'
    args.train_file = 'train.h5'
    args.valid_file = 'valid.h5'
    args.test_file = 'test.h5'
    args.output_dir = str(tmp_path)
    args.batch_max_edges = 16
    args.batch_max_nodes = None
    args.epochs = 0
    args.tqdm = False

    bundle = jax_backend.ModelBundle(
        config={'atomic_numbers': [1], 'r_max': 3.0},
        params={'w': jnp.asarray(1.0, dtype=jnp.float32)},
        module=object(),
    )
    calls = []

    class Logger:
        def log(self, *_args, **_kwargs):
            pass

    class Optimizer:
        def init(self, _params):
            return {}

    class Scheduler:
        current_lr = 0.1
        monitor = 'val'

        def register_initial_metric(self, *_args, **_kwargs):
            pass

        def update_after_epoch(self, *_args, **_kwargs):
            return False

    def fake_get_dataloader(*_args, data_file, **_kwargs):
        return [data_file]

    def fake_run_eval_loop(
        _params,
        loader,
        _eval_step_fn,
        *,
        max_steps,
        multi_device,
        logger=None,
    ):
        calls.append((list(loader), max_steps, multi_device))
        return None, jax_backend.JaxLossCollection()

    monkeypatch.setattr(jax_backend, '_launch_local_processes', lambda _args: None)
    monkeypatch.setattr(
        jax_backend, 'validate_training_args', lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(jax_backend, 'set_jax_platform', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(jax_backend, '_initialize_distributed', lambda _args: None)
    monkeypatch.setattr(jax_backend, '_shutdown_distributed', lambda: None)
    monkeypatch.setattr(
        jax_backend, 'ensure_output_dir', lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(jax_backend, 'init_logger', lambda *_args, **_kwargs: Logger())
    monkeypatch.setattr(
        jax_backend, 'load_model_bundle', lambda *_args, **_kwargs: bundle
    )
    monkeypatch.setattr(jax_backend, 'get_dataloader', fake_get_dataloader)
    monkeypatch.setattr(
        jax_backend, 'create_wrapper', lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        jax_backend, 'make_apply_fn', lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        jax_backend, 'build_loss_fn', lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        jax_backend, 'build_trainable_mask', lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        jax_backend, 'optimizer_kwargs', lambda _args: {'learning_rate': 0.1}
    )
    monkeypatch.setattr(jax_backend, 'create_optimizer', lambda **_kwargs: Optimizer())
    monkeypatch.setattr(
        jax_backend,
        'create_scheduler_controller',
        lambda *_args, **_kwargs: Scheduler(),
    )
    monkeypatch.setattr(
        jax_backend.jax_checkpoint,
        'load_checkpoint',
        lambda _args, _bundle, opt_state, _logger: (_bundle, opt_state, None, None),
    )
    monkeypatch.setattr(
        jax_backend,
        '_build_train_functions',
        lambda *_args, **_kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        jax_backend, '_build_eval_step', lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(jax_backend, '_run_eval_loop', fake_run_eval_loop)
    monkeypatch.setattr(jax_backend, '_is_multi_device', lambda: False)
    monkeypatch.setattr(jax_backend.jax, 'process_count', lambda: 2)
    monkeypatch.setattr(jax_backend.jax, 'process_index', lambda: 1)

    from jax.experimental import multihost_utils

    monkeypatch.setattr(multihost_utils, 'sync_global_devices', lambda *_args: None)

    summary = jax_backend.train(args)

    assert summary['test_loss'] is None
    assert calls[-1][0] == ['test.h5']


def test_shard_map_eval_step_uses_replicated_params_and_outputs():
    device_count = jax.local_device_count()
    params = {'w': jnp.array([1.0, 2.0], dtype=jnp.float32)}
    batch = {
        'x': jnp.arange(device_count, dtype=jnp.float32)[:, None]
        + jnp.array([3.0, 4.0], dtype=jnp.float32)
    }

    def loss_fn(local_params, local_batch):
        assert local_params['w'].shape == (2,)
        assert local_batch['x'].shape == (2,)
        loss = jnp.sum(local_params['w'] * local_batch['x'])
        return loss, {'total': loss}

    eval_step = _build_eval_step(loss_fn, multi_device=True)

    loss, aux = eval_step(params, batch)

    expected_loss = np.mean(
        np.sum(
            np.asarray(batch['x']) * np.asarray(params['w']),
            axis=1,
        )
    )
    assert loss.shape == ()
    assert aux['total'].shape == ()
    np.testing.assert_allclose(np.asarray(loss), expected_loss)


def test_shard_map_train_step_updates_state_without_tuple_wrapper():
    device_count = jax.local_device_count()
    base_params = {'w': jnp.array([1.0, 2.0], dtype=jnp.float32)}
    state = TrainState(
        params=base_params,
        opt_state=optax.sgd(0.1).init(base_params),
        ema_params=None,
    )
    batch = {
        'x': jnp.arange(device_count, dtype=jnp.float32)[:, None]
        + jnp.array([3.0, 4.0], dtype=jnp.float32)
    }

    def loss_fn(local_params, local_batch):
        assert local_params['w'].shape == (2,)
        assert local_batch['x'].shape == (2,)
        loss = jnp.sum(local_params['w'] * local_batch['x'])
        return loss, {'total': loss}

    grad_step, apply_updates = _build_train_functions(
        loss_fn,
        optax.sgd(0.1),
        grad_clip_value=None,
        use_ema=False,
        multi_device=True,
    )

    loss, aux, grads = grad_step(state.params, batch)
    updated_state = apply_updates(state, grads, 0.0)

    expected_grad = np.mean(np.asarray(batch['x']), axis=0)
    expected_params = np.asarray(base_params['w']) - 0.1 * expected_grad

    assert loss.shape == ()
    assert aux['total'].shape == ()
    assert grads['w'].shape == (2,)
    np.testing.assert_allclose(np.asarray(grads['w']), expected_grad)
    assert isinstance(updated_state, TrainState)
    np.testing.assert_allclose(
        np.asarray(updated_state.params['w']),
        expected_params,
    )


def test_shard_map_train_step_ignores_zero_count_metric_shards():
    if jax.local_device_count() < 2:
        pytest.skip('count-weighted gradient test requires multiple local devices')

    device_count = jax.local_device_count()
    base_params = {'w': jnp.array([1.0, 2.0], dtype=jnp.float32)}
    state = TrainState(
        params=base_params,
        opt_state=optax.sgd(0.1).init(base_params),
        ema_params=None,
    )
    x = jnp.arange(device_count, dtype=jnp.float32)[:, None] + jnp.array(
        [3.0, 4.0], dtype=jnp.float32
    )
    counts = jnp.zeros((device_count,), dtype=jnp.float32).at[0].set(1.0)
    batch = {'x': x, 'count': counts}

    def loss_fn(local_params, local_batch):
        loss = jnp.sum(local_params['w'] * local_batch['x'])
        count = local_batch['count']
        return loss, {
            'metrics': {
                'total': (loss, count),
                'energy': (loss, count),
                'forces': (jnp.asarray(0.0, dtype=jnp.float32), count * 0.0),
                'stress': (jnp.asarray(0.0, dtype=jnp.float32), count * 0.0),
            },
            'per_graph_error': jnp.asarray([loss], dtype=jnp.float32),
        }

    grad_step, apply_updates = _build_train_functions(
        loss_fn,
        optax.sgd(0.1),
        grad_clip_value=None,
        use_ema=False,
        multi_device=True,
    )

    loss, aux, grads = grad_step(state.params, batch)
    updated_state = apply_updates(state, grads, 0.0)

    expected_grad = np.asarray(batch['x'][0])
    expected_loss = float(np.sum(np.asarray(base_params['w']) * expected_grad))
    expected_params = np.asarray(base_params['w']) - 0.1 * expected_grad

    np.testing.assert_allclose(np.asarray(loss), expected_loss)
    np.testing.assert_allclose(np.asarray(aux['metrics']['total'][1]), 1.0)
    np.testing.assert_allclose(np.asarray(grads['w']), expected_grad)
    np.testing.assert_allclose(np.asarray(updated_state.params['w']), expected_params)


def test_shard_map_train_step_preserves_equal_weight_for_nonempty_metric_shards():
    if jax.local_device_count() < 2:
        pytest.skip('non-empty shard averaging test requires multiple local devices')

    device_count = jax.local_device_count()
    base_params = {'w': jnp.array([1.0, 2.0], dtype=jnp.float32)}
    state = TrainState(
        params=base_params,
        opt_state=optax.sgd(0.1).init(base_params),
        ema_params=None,
    )
    x = jnp.arange(device_count, dtype=jnp.float32)[:, None] + jnp.array(
        [3.0, 4.0], dtype=jnp.float32
    )
    counts = jnp.arange(1, device_count + 1, dtype=jnp.float32)
    batch = {'x': x, 'count': counts}

    def loss_fn(local_params, local_batch):
        loss = jnp.sum(local_params['w'] * local_batch['x'])
        count = local_batch['count']
        return loss, {
            'metrics': {
                'total': (loss, count),
                'energy': (loss, count),
                'forces': (jnp.asarray(0.0, dtype=jnp.float32), count * 0.0),
                'stress': (jnp.asarray(0.0, dtype=jnp.float32), count * 0.0),
            },
            'per_graph_error': jnp.asarray([loss], dtype=jnp.float32),
        }

    grad_step, apply_updates = _build_train_functions(
        loss_fn,
        optax.sgd(0.1),
        grad_clip_value=None,
        use_ema=False,
        multi_device=True,
    )

    _, aux, grads = grad_step(state.params, batch)
    updated_state = apply_updates(state, grads, 0.0)

    expected_grad = np.mean(np.asarray(batch['x']), axis=0)
    expected_metric_loss = np.average(
        np.sum(np.asarray(batch['x']) * np.asarray(base_params['w']), axis=1),
        weights=np.asarray(batch['count']),
    )
    expected_params = np.asarray(base_params['w']) - 0.1 * expected_grad

    np.testing.assert_allclose(
        np.asarray(aux['metrics']['total'][0]), expected_metric_loss
    )
    np.testing.assert_allclose(np.asarray(grads['w']), expected_grad)
    np.testing.assert_allclose(np.asarray(updated_state.params['w']), expected_params)


def test_predict_streaming_single_device_does_not_require_shard_map(monkeypatch):
    if jax.local_device_count() != 1:
        pytest.skip('single-device prediction path requires exactly one local device')

    def fail_shard_map(*_args, **_kwargs):
        raise AssertionError('single-device prediction should use jax.jit')

    monkeypatch.setattr(jax_predict, 'shard_map_over_local_devices', fail_shard_map)

    graph = jraph.GraphsTuple(
        nodes={'positions': jnp.zeros((2, 3), dtype=jnp.float32)},
        edges={'shifts': jnp.zeros((0, 3), dtype=jnp.float32)},
        senders=jnp.zeros((0,), dtype=jnp.int32),
        receivers=jnp.zeros((0,), dtype=jnp.int32),
        globals={
            'cell': jnp.zeros((3, 3, 3), dtype=jnp.float32),
            'graph_id': jnp.array([0, 1, -1], dtype=jnp.int32),
        },
        n_node=jnp.array([1, 1, 0], dtype=jnp.int32),
        n_edge=jnp.array([0, 0, 0], dtype=jnp.int32),
    )

    class Loader:
        def iter_batches(self, **_kwargs):
            yield graph

    def predictor(params, batch):
        return {
            'energy': jnp.full(
                (batch.n_node.shape[0],), params['bias'], dtype=jnp.float32
            )
        }

    graph_ids, outputs = predict_streaming(
        predictor,
        {'bias': jnp.asarray(3.0, dtype=jnp.float32)},
        Loader(),
        progress_bar=False,
        device_prefetch_batches=0,
    )

    assert graph_ids == [0]
    np.testing.assert_allclose(np.asarray(outputs['energy']), [3.0])


def test_predict_streaming_single_device_accepts_prepared_device_batch():
    if jax.local_device_count() != 1:
        pytest.skip('single-device pre-sharded path requires exactly one local device')

    graph = jraph.GraphsTuple(
        nodes={'positions': jnp.zeros((2, 3), dtype=jnp.float32)},
        edges={'shifts': jnp.zeros((0, 3), dtype=jnp.float32)},
        senders=jnp.zeros((0,), dtype=jnp.int32),
        receivers=jnp.zeros((0,), dtype=jnp.int32),
        globals={
            'cell': jnp.zeros((3, 3, 3), dtype=jnp.float32),
            'graph_id': jnp.array([0, 1, -1], dtype=jnp.int32),
        },
        n_node=jnp.array([1, 1, 0], dtype=jnp.int32),
        n_edge=jnp.array([0, 0, 0], dtype=jnp.int32),
    )
    device_batch = jax.tree_util.tree_map(lambda x: x[None, ...], graph)

    class Loader:
        def iter_batches(self, **_kwargs):
            yield device_batch

    def predictor(params, batch):
        assert batch.n_node.ndim == 1
        assert batch.nodes['positions'].ndim == 2
        return {
            'energy': jnp.full(
                (batch.n_node.shape[0],), params['bias'], dtype=jnp.float32
            )
        }

    graph_ids, outputs = predict_streaming(
        predictor,
        {'bias': jnp.asarray(4.0, dtype=jnp.float32)},
        Loader(),
        progress_bar=False,
        device_prefetch_batches=0,
    )

    assert graph_ids == [0]
    np.testing.assert_allclose(np.asarray(outputs['energy']), [4.0])


def test_shard_graphs_for_devices_keeps_equal_shape_chunk_graphs():
    graph = jraph.GraphsTuple(
        nodes={'positions': jnp.zeros((4, 3), dtype=jnp.float32)},
        edges={'shifts': jnp.zeros((0, 3), dtype=jnp.float32)},
        senders=jnp.zeros((0,), dtype=jnp.int32),
        receivers=jnp.zeros((0,), dtype=jnp.int32),
        globals={
            'cell': jnp.zeros((4, 3, 3), dtype=jnp.float32),
            'graph_id': jnp.array([0, 1, 2, 3], dtype=jnp.int32),
        },
        n_node=jnp.array([1, 1, 1, 1], dtype=jnp.int32),
        n_edge=jnp.array([0, 0, 0, 0], dtype=jnp.int32),
    )

    chunks = shard_graphs_for_devices(graph, 2)
    kept_ids = []
    for chunk in chunks:
        mask = np.asarray(jraph.get_graph_padding_mask(chunk), dtype=bool)
        graph_ids = np.asarray(chunk.globals['graph_id'])
        kept_ids.extend(int(value) for value in graph_ids[mask] if value >= 0)

    assert kept_ids == [0, 1, 2, 3]


def test_predict_streaming_accepts_prepared_device_batch():
    if jax.local_device_count() < 2:
        pytest.skip('pre-sharded prediction path requires multiple local devices')

    graph = jraph.GraphsTuple(
        nodes={'positions': jnp.zeros((4, 3), dtype=jnp.float32)},
        edges={'shifts': jnp.zeros((0, 3), dtype=jnp.float32)},
        senders=jnp.zeros((0,), dtype=jnp.int32),
        receivers=jnp.zeros((0,), dtype=jnp.int32),
        globals={
            'cell': jnp.zeros((4, 3, 3), dtype=jnp.float32),
            'graph_id': jnp.array([0, 1, 2, 3], dtype=jnp.int32),
        },
        n_node=jnp.array([1, 1, 1, 1], dtype=jnp.int32),
        n_edge=jnp.array([0, 0, 0, 0], dtype=jnp.int32),
    )
    device_batch = prepare_sharded_batch(graph, jax.local_device_count())

    class Loader:
        def iter_batches(self, **_kwargs):
            yield device_batch

    def predictor(params, batch):
        return {
            'energy': jnp.full(
                (batch.n_node.shape[0],), params['bias'], dtype=jnp.float32
            )
        }

    graph_ids, outputs = predict_streaming(
        predictor,
        {'bias': jnp.asarray(7.0, dtype=jnp.float32)},
        Loader(),
        progress_bar=False,
        device_prefetch_batches=0,
    )

    assert graph_ids == [0, 1, 2, 3]
    np.testing.assert_allclose(np.asarray(outputs['energy']), [7.0, 7.0, 7.0, 7.0])


def test_shard_map_predict_streaming_squeezes_graph_batch_axis():
    graph = jraph.GraphsTuple(
        nodes={'positions': jnp.zeros((2, 3), dtype=jnp.float32)},
        edges={'shifts': jnp.zeros((0, 3), dtype=jnp.float32)},
        senders=jnp.zeros((0,), dtype=jnp.int32),
        receivers=jnp.zeros((0,), dtype=jnp.int32),
        globals={
            'cell': jnp.zeros((3, 3, 3), dtype=jnp.float32),
            'graph_id': jnp.array([0, 1, -1], dtype=jnp.int32),
        },
        n_node=jnp.array([1, 1, 0], dtype=jnp.int32),
        n_edge=jnp.array([0, 0, 0], dtype=jnp.int32),
    )

    class Loader:
        def iter_batches(self, **_kwargs):
            yield graph

    def predictor(params, batch):
        assert batch.n_node.ndim == 1
        assert batch.nodes['positions'].ndim == 2
        return {
            'energy': jnp.full(
                (batch.n_node.shape[0],), params['bias'], dtype=jnp.float32
            )
        }

    graph_ids, outputs = predict_streaming(
        predictor,
        {'bias': jnp.asarray(5.0, dtype=jnp.float32)},
        Loader(),
        progress_bar=False,
        device_prefetch_batches=0,
    )

    if jax.local_device_count() > 1:
        assert graph_ids == [0, 1]
        np.testing.assert_allclose(np.asarray(outputs['energy']), [5.0, 5.0])
    else:
        assert graph_ids == [0]
        np.testing.assert_allclose(np.asarray(outputs['energy']), [5.0])
