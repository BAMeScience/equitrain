from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip(
    'jax', reason='JAX runtime is required for JAX backend tests.'
)
jraph = pytest.importorskip('jraph', reason='jraph is required for JAX backend tests.')
optax = pytest.importorskip('optax', reason='optax is required for JAX backend tests.')

import jax.numpy as jnp

from equitrain.backends import jax_predict
from equitrain.backends.jax_backend import (
    TrainState,
    _build_eval_step,
    _build_train_functions,
)
from equitrain.backends.jax_predict import predict_streaming
from equitrain.backends.jax_utils import (
    prepare_sharded_batch,
    shard_graphs_for_devices,
)


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
