"""Serial equivalence and boundary cases for global-mesh JAX training."""

import os
import pickle
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jraph = pytest.importorskip('jraph')
optax = pytest.importorskip('optax')

import jax.numpy as jnp

from equitrain.backends import jax_backend, jax_checkpoint, jax_distributed, jax_utils
from equitrain.backends.jax_loss import JaxLossCollection, update_collection_from_aux
from equitrain.backends.jax_loss_fn import LossSettings, build_loss_fn
from equitrain.data.backend_jax.atoms_to_graphs_impl import _AttrDict


def _graph(atoms, target):
    positions = np.arange(atoms * 3, dtype=np.float32).reshape(atoms, 3) / 10 + 0.2
    return jraph.GraphsTuple(
        nodes=_AttrDict(positions=positions, forces=np.full_like(positions, target)),
        edges=_AttrDict(shifts=np.zeros((atoms, 3), np.float32)),
        senders=np.arange(atoms, dtype=np.int32),
        receivers=np.roll(np.arange(atoms, dtype=np.int32), 1),
        globals=_AttrDict(
            energy=np.asarray([target], np.float32),
            stress=np.full((1, 3, 3), target, np.float32),
            cell=np.eye(3, dtype=np.float32)[None],
            weight=np.ones(1, np.float32),
        ),
        n_node=np.asarray([atoms], np.int32),
        n_edge=np.asarray([atoms], np.int32),
    )


def _batch(graphs):
    return jraph.pad_with_graphs(jraph.batch(graphs), n_node=16, n_edge=16, n_graph=5)


def _predict(params, graph):
    def energy(positions):
        node_energy = jnp.sum(
            params['w'][0] * positions**2 + params['w'][1] * positions, axis=1
        )
        ids = jnp.repeat(
            jnp.arange(len(graph.n_node)),
            graph.n_node,
            total_repeat_length=len(positions),
        )
        return jraph.segment_sum(node_energy, ids, len(graph.n_node))

    # Exercise differentiation through forces, as in atomistic models.
    forces = -jax.grad(lambda positions: jnp.sum(energy(positions)))(
        graph.nodes.positions
    )
    return {
        'energy': energy(graph.nodes.positions),
        'forces': forces,
        'stress': params['w'][2] * graph.globals.cell,
    }


def _loss():
    return build_loss_fn(
        _predict,
        LossSettings(
            energy_weight=0.7,
            forces_weight=1.3,
            stress_weight=0.2,
            loss_type_energy='mse',
            loss_type_forces='mse',
            loss_type_stress='mse',
        ),
    )


def _params():
    return {'w': jnp.asarray([0.4, -0.2, 0.8], jnp.float32)}


def _device_batch(graphs):
    count = jax.local_device_count()
    graphs = list(graphs)
    graphs.extend(
        jax_distributed.empty_graph_like(graphs[0]) for _ in range(count - len(graphs))
    )
    return jax_utils.process_local_sharded_to_global(
        jax_utils.prepare_sharded_batch(graphs, count)
    )


def _assert_tree_close(actual, expected):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(
        expected
    )
    for a, b in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        assert a.shape == b.shape
        np.testing.assert_allclose(a, b, rtol=3e-6, atol=2e-6)


def test_mixed_component_gradients_and_metrics_match_serial():
    if jax.local_device_count() < 2:
        pytest.skip('requires at least two local devices')
    graphs = [_graph(1, 1.0), _graph(2, -2.0), _graph(3, 3.0)]
    params, loss_fn = _params(), _loss()
    (expected_loss, expected_aux), expected_grad = jax.jit(
        jax.value_and_grad(loss_fn, has_aux=True)
    )(params, _batch(graphs))
    grad_step, _ = jax_backend._build_train_functions(
        loss_fn, optax.sgd(0.01), grad_clip_value=None, use_ema=False, multi_device=True
    )
    loss, aux, grad = grad_step(
        params, _device_batch([_batch(graphs[:1]), _batch(graphs[1:])])
    )
    np.testing.assert_allclose(loss, expected_loss, rtol=2e-6)
    _assert_tree_close(grad, expected_grad)
    _assert_tree_close(aux['metrics'], expected_aux['metrics'])
    assert aux['per_graph_error'].shape == (5 * jax.local_device_count(),)
    eval_step = jax_backend._build_eval_step(loss_fn, multi_device=True)
    eval_loss, eval_aux = eval_step(
        params, _device_batch([_batch(graphs[:1]), _batch(graphs[1:])])
    )
    np.testing.assert_allclose(eval_loss, expected_loss, rtol=2e-6)
    _assert_tree_close(eval_aux['metrics'], expected_aux['metrics'])


def test_epoch_total_uses_each_components_denominator():
    graphs = [_graph(1, 1.0), _graph(2, -2.0), _graph(3, 3.0)]
    loss_fn, collection = _loss(), JaxLossCollection()
    expected, expected_aux = loss_fn(_params(), _batch(graphs))
    for group in (graphs[:1], graphs[1:]):
        _, aux = loss_fn(_params(), _batch(group))
        update_collection_from_aux(collection, aux)
    assert collection.mean() == pytest.approx(float(expected), rel=2e-6)
    for name, (value, count) in expected_aux['metrics'].items():
        assert collection.components[name].value == pytest.approx(
            float(value), rel=2e-6
        )
        assert collection.components[name].count == float(count)


def test_gradients_are_clipped_after_global_reduction():
    if jax.local_device_count() < 2:
        pytest.skip('requires at least two local devices')
    count = jax.local_device_count()
    slopes = jnp.asarray([10.0, -8.0] + [0.0] * (count - 2))

    def loss_fn(p, batch):
        value = p * batch['slope']
        return value, {'metrics': {'total': (value, batch['count'])}}

    grad_step, _ = jax_backend._build_train_functions(
        loss_fn, optax.sgd(0.1), grad_clip_value=2.0, use_ema=False, multi_device=True
    )
    _, _, grad = grad_step(
        jnp.asarray(0.0),
        {'slope': slopes, 'count': jnp.asarray([1.0, 1.0] + [0.0] * (count - 2))},
    )
    assert float(grad) == pytest.approx(1.0)


@pytest.mark.parametrize('multi_device', [False, True])
def test_fused_step_skips_empty_updates_and_preserves_freezing(multi_device):
    params, optimizer = _params(), optax.adamw(0.1, weight_decay=0.2)
    state = jax_backend.TrainState(params, optimizer.init(params), params)
    steps = jax_backend._build_train_functions(
        _loss(),
        optimizer,
        grad_clip_value=None,
        use_ema=True,
        multi_device=multi_device,
    )
    mask = {'w': np.asarray([True, False, True])}
    step = jax_backend._build_train_step(*steps, use_ema=True, ema_decay=0.9, mask=mask)
    graph = _batch([_graph(2, 2.0)])
    prepare = (
        (lambda g: _device_batch([g]))
        if multi_device
        else jax_utils.prepare_single_batch
    )
    batch, empty = prepare(graph), prepare(jax_distributed.empty_graph_like(graph))
    if multi_device:
        state = jax_utils.replicate_global(state)

    # Nonzero Adam moments make accidental empty updates observable as well as decay.
    after_valid, _, count = step(state, batch, 0)
    after_empty, _, empty_count = step(after_valid, empty, count)
    _assert_tree_close(after_empty, after_valid)
    assert int(empty_count) == int(count) == 1
    expected, _, expected_count = step(after_valid, batch, count)
    actual, _, actual_count = step(after_empty, batch, empty_count)
    _assert_tree_close(actual, expected)
    assert int(actual_count) == int(expected_count) == 2
    np.testing.assert_array_equal(actual.params['w'][1], params['w'][1])
    np.testing.assert_array_equal(actual.ema_params['w'][1], params['w'][1])
    assert not np.array_equal(actual.params['w'], params['w'])


@pytest.mark.parametrize('placement', ['host', 'default', 'committed'])
@pytest.mark.parametrize('frozen', [False, True])
def test_checkpoint_roundtrip_preserves_replicated_parameter_dimensions(
    tmp_path, placement, frozen
):
    from flax.core import freeze

    shape = (jax.local_device_count(), 3)
    params = {'zero': np.zeros(shape, np.float32), 'one': np.ones(shape, np.float32)}
    if frozen:
        params = freeze(params)
    if placement == 'default':
        params = jax.device_put(params)
    elif placement == 'committed':
        params = jax.device_put(params, jax.local_devices()[0])
    opt_state = optax.adam(0.1).init(params)
    state = jax_utils.replicate_global(
        jax_backend.TrainState(params, opt_state, params)
    )
    host = jax_backend._unreplicate(state)
    _assert_tree_close(host, jax_backend.TrainState(params, opt_state, params))
    # A tuple with one entry per device is also an ordinary model structure.
    values = tuple(np.ones(shape, np.float32) for _ in range(jax.local_device_count()))
    _assert_tree_close(jax_backend._unreplicate(values), values)

    args = SimpleNamespace(output_dir=str(tmp_path), keep_best_checkpoints=0)
    metric = SimpleNamespace(
        main=SimpleNamespace(meters={'total': SimpleNamespace(avg=1.0)})
    )
    bundle = jax_utils.ModelBundle(config={}, params=host.params, module=None)
    jax_checkpoint.save_checkpoint(
        args, 1, metric, bundle, host.opt_state, ema_params=host.ema_params
    )
    args.load_checkpoint = str(next(tmp_path.glob('best_*')))
    restored, restored_opt, _, restored_ema = jax_checkpoint.load_checkpoint(
        args, bundle, opt_state
    )
    for key in params:
        np.testing.assert_array_equal(restored.params[key], params[key])
        np.testing.assert_array_equal(restored_ema[key], params[key])
    _assert_tree_close(restored_opt, opt_state)
    reloaded = jax_backend.TrainState(restored.params, restored_opt, restored_ema)
    assert jax.tree_util.tree_structure(reloaded) == jax.tree_util.tree_structure(host)

    # A checkpoint must support another update, not only recover equal values.
    optimizer = optax.adam(0.1)

    def advance(state):
        gradients = jax.tree_util.tree_map(jnp.ones_like, state.params)
        updates, optimizer_state = optimizer.update(
            gradients, state.opt_state, state.params
        )
        return optax.apply_updates(state.params, updates), optimizer_state

    _assert_tree_close(jax.jit(advance)(reloaded), jax.jit(advance)(host))


def test_legacy_replication_requires_explicit_device_axis():
    original = {'w': np.ones((jax.local_device_count(), 3), np.float32)}
    replicated = jax_utils.replicate_to_local_devices(original)
    _assert_tree_close(
        jax_utils.unreplicate_from_local_devices(replicated, replicated_axis=True),
        original,
    )


@pytest.mark.parametrize('explicit_mask', [False, True])
@pytest.mark.parametrize('multi_device', [False, True])
def test_fused_step_preserves_mace_configuration(explicit_mask, multi_device):
    config_module = pytest.importorskip('mace_jax.nnx_config')
    params = {
        'w': jnp.asarray(1.0),
        'config': config_module.ConfigDict(
            {'indices': jnp.arange(3), 'scale': jnp.asarray(2.0)}
        ),
        'indices': jnp.arange(3),
        'flags': jnp.asarray([True, False]),
    }
    mask = (
        {'w': True, 'config': False, 'indices': False, 'flags': False}
        if explicit_mask
        else None
    )
    optimizer = optax.adamw(0.1, weight_decay=0.2)
    if mask is not None:
        optimizer = optax.masked(optimizer, mask)

    def loss_fn(p, batch):
        value = (p['w'] * p['config'].get('scale') - batch['target']) ** 2
        return value, {'metrics': {'total': (value, batch['count'])}}

    steps = jax_backend._build_train_functions(
        loss_fn,
        optimizer,
        grad_clip_value=None,
        use_ema=True,
        multi_device=multi_device,
    )
    step = jax_backend._build_train_step(*steps, use_ema=True, ema_decay=0.9, mask=mask)
    state = jax_utils.replicate_global(
        jax_backend.TrainState(params, optimizer.init(params), params)
    )
    shape = (jax.local_device_count(),) if multi_device else ()
    batch = {'target': jnp.zeros(shape), 'count': jnp.ones(shape)}
    updated, _, count = step(state, batch, 0)
    for name in ('config', 'indices', 'flags'):
        _assert_tree_close(updated.params[name], params[name])
        _assert_tree_close(updated.ema_params[name], params[name])
    for before, after in zip(
        jax.tree_util.tree_leaves(state),
        jax.tree_util.tree_leaves(updated),
        strict=True,
    ):
        assert before.dtype == after.dtype
    assert float(updated.params['w']) < 1
    assert int(count) == 1
    after_empty, _, empty_count = step(
        updated, dict(batch, count=jnp.zeros(shape)), count
    )
    _assert_tree_close(after_empty, updated)
    assert int(empty_count) == 1


@pytest.mark.parametrize('multi_device', [False, True])
@pytest.mark.parametrize('unfreeze', [[], ['.*']])
def test_nnx_float_configuration_is_never_trainable(multi_device, unfreeze):
    pytest.importorskip('mace_jax')
    from flax import nnx
    from mace_jax.nnx_config import ConfigVar
    from mace_jax.nnx_utils import state_to_pure_dict

    from equitrain.backends.jax_freeze import build_trainable_mask

    class Model(nnx.Module):
        def __init__(self):
            self.w = nnx.Param(jnp.asarray(1.0))
            self.scale = ConfigVar(jnp.asarray(2.0))
            self.constant = ConfigVar(jnp.asarray(3.0))

    module, variables = nnx.split(Model())
    params = state_to_pure_dict(variables)
    mask = build_trainable_mask(
        SimpleNamespace(unfreeze_params=unfreeze, freeze_params=[]),
        params,
        module=module,
    )
    assert mask == {'w': True, 'scale': False, 'constant': False}
    optimizer = optax.adamw(0.1, weight_decay=0.2)
    user_mask = build_trainable_mask(
        SimpleNamespace(unfreeze_params=unfreeze, freeze_params=[]), params
    )
    if user_mask is not None:
        optimizer = optax.masked(optimizer, user_mask)

    def loss_fn(p, batch):
        value = (p['w'] * p['scale']) ** 2
        return value, {'metrics': {'total': (value, batch['count'])}}

    functions = jax_backend._build_train_functions(
        loss_fn,
        optimizer,
        grad_clip_value=None,
        use_ema=True,
        multi_device=multi_device,
    )
    step = jax_backend._build_train_step(
        *functions, use_ema=True, ema_decay=0.9, mask=mask
    )
    state = jax_utils.replicate_global(
        jax_backend.TrainState(params, optimizer.init(params), params)
    )
    shape = (jax.local_device_count(),) if multi_device else ()
    updated, _, _ = step(state, {'count': jnp.ones(shape)}, 0)
    for name in ('scale', 'constant'):
        np.testing.assert_array_equal(updated.params[name], params[name])
        np.testing.assert_array_equal(updated.ema_params[name], params[name])
    assert float(updated.params['w']) < 1.0


def test_small_dataset_and_partial_tail_keep_geometry():
    graph = _batch([_graph(2, 1.0)])
    chunks = list(
        jax_distributed.coordinated_chunks([graph] * 3, 2, phase='test', logger=None)
    )
    assert [len(c) for c in chunks] == [2, 2]
    dummy = chunks[-1][-1]
    _assert_tree_close(dummy.nodes, graph.nodes)
    _assert_tree_close(dummy.edges, graph.edges)
    np.testing.assert_array_equal(dummy.n_node, graph.n_node)
    np.testing.assert_array_equal(dummy.senders, graph.senders)
    assert np.sum(dummy.n_node) == len(dummy.nodes.positions)
    assert not np.any(dummy.globals.weight)
    loss, aux = _loss()(_params(), dummy)
    assert float(loss) == 0
    assert float(aux['metrics']['total'][1]) == 0
    assert (
        len(
            list(
                jax_distributed.coordinated_chunks(
                    [graph], 2, phase='test', logger=None
                )
            )
        )
        == 1
    )


def test_initially_empty_rank_receives_padding_template(monkeypatch):
    graph = _batch([_graph(2, 1.0)])
    states = [
        np.stack([np.zeros(9, np.int32), np.r_[1, jax_distributed._signature(graph)]]),
        np.zeros((2, 9), np.int32),
    ]
    monkeypatch.setattr(jax, 'process_count', lambda: 2)
    monkeypatch.setattr(jax, 'process_index', lambda: 0)
    monkeypatch.setattr(
        jax_distributed.multihost_utils,
        'process_allgather',
        lambda *_a, **_k: states.pop(0),
    )

    payload = pickle.dumps(jax.device_get(graph))
    messages = [np.asarray(len(payload), np.int32), np.frombuffer(payload, np.uint8)]

    def broadcast(value, *, is_source):
        assert not is_source
        return messages.pop(0)

    monkeypatch.setattr(
        jax_distributed.multihost_utils, 'broadcast_one_to_all', broadcast
    )
    chunks = list(
        jax_distributed.coordinated_chunks(
            [], 1, phase='test', logger=None, sync_processes=True
        )
    )
    assert not messages
    assert len(chunks) == 1
    assert not np.any(chunks[0][0].globals.weight)
    _assert_tree_close(chunks[0][0].nodes, graph.nodes)


@pytest.mark.parametrize('failure', ['local', 'remote', 'shape', 'eager'])
def test_rank_failures_are_coordinated(monkeypatch, failure):
    graph = _batch([_graph(2, 1.0)])
    loader = [graph]
    if failure in {'local', 'eager'}:

        class BrokenLoader:
            def iter_batches(self, **kwargs):
                raise OSError('broken data')

            def __iter__(self):
                raise OSError('broken data')

        loader = BrokenLoader()
        if failure == 'eager':
            loader = jax_backend._iter_loader_for_epoch(
                loader, epoch=1, seed=0, process_count=2, process_index=0
            )

    def gather(state, **kwargs):
        other = state.copy()
        if failure == 'remote':
            other[0] = -1
        elif failure == 'shape':
            other[1] ^= 1
        return np.stack([state, other])

    monkeypatch.setattr(jax_distributed.multihost_utils, 'process_allgather', gather)
    expected_error = ValueError if failure == 'shape' else RuntimeError
    with pytest.raises(
        expected_error, match='matching fixed batch|Batch loading failed'
    ):
        list(
            jax_distributed.coordinated_chunks(
                loader, 1, phase='test', logger=None, sync_processes=True
            )
        )


def test_train_epoch_keeps_tail_batch():
    params = _params()
    optimizer = optax.sgd(0.01)
    state = jax_backend.TrainState(params, optimizer.init(params), None)
    steps = jax_backend._build_train_functions(
        _loss(), optimizer, grad_clip_value=None, use_ema=False, multi_device=True
    )
    train_step = jax_backend._build_train_step(*steps, use_ema=False, ema_decay=None)
    graph = _batch([_graph(2, 1.0)])
    count = jax.local_device_count()
    # Identical graphs give the same serial gradient in the full and tail batches.
    expected = state
    for _ in range(2):
        grads = jax.grad(lambda p: _loss()(p, graph)[0])(expected.params)
        updates, opt_state = optimizer.update(
            grads, expected.opt_state, expected.params
        )
        expected = expected.replace(
            params=optax.apply_updates(expected.params, updates), opt_state=opt_state
        )
    actual, metrics, ema_count = jax_backend._run_train_epoch(
        jax_utils.replicate_global(state),
        [graph] * (count + 1),
        train_step,
        max_steps=None,
        multi_device=True,
        ema_count_start=0,
        logger=None,
        args=SimpleNamespace(tqdm=False, verbose=0),
        epoch=1,
        metric_settings=dict(
            include_energy=True,
            include_forces=True,
            include_stress=True,
            loss_label='mse',
        ),
        learning_rate=0.01,
        is_primary=True,
    )
    _assert_tree_close(actual, expected)
    assert ema_count == 0
    assert metrics.components['total'].count == count + 1


@pytest.mark.parametrize('include_configuration', [False, True])
def test_mace_force_training_matches_serial_update(include_configuration):
    pytest.importorskip('mace_jax')
    from flax import nnx
    from flax.traverse_util import flatten_dict
    from mace_jax.tools import gin_model, model_builder

    from equitrain.backends.jax_freeze import build_trainable_mask

    graph = _graph(2, 1.0)
    graph.nodes['species'] = np.zeros(2, np.int32)
    graph.edges['unit_shifts'] = np.zeros((2, 3), np.float32)
    graph = jraph.pad_with_graphs(graph, n_node=3, n_edge=3, n_graph=2)
    predictor, bundle, _ = gin_model.model(
        r_max=3.0,
        num_species=1,
        atomic_numbers=(1,),
        atomic_energies='zero',
        avg_num_neighbors=1.0,
        num_interactions=2,
        num_bessel=2,
        num_polynomial_cutoff=5,
        max_ell=1,
        correlation=2,
        gate=jax.nn.silu,
        hidden_irreps='2x0e+2x1o',
        MLP_irreps='2x0e',
    )
    params = bundle['params']
    loss_fn = build_loss_fn(
        predictor,
        LossSettings(
            energy_weight=1.0,
            forces_weight=1.0,
            stress_weight=0.0,
            loss_type_energy='mse',
            loss_type_forces='mse',
        ),
    )
    expected_loss, grads = jax.jit(jax.value_and_grad(lambda p, g: loss_fn(p, g)[0]))(
        params, graph
    )
    assert np.isfinite(float(optax.global_norm(grads)))
    assert float(optax.global_norm(grads)) > 0
    optimizer = optax.sgd(0.001)
    state = jax_backend.TrainState(params, optimizer.init(params), None)
    updates, expected_opt = optimizer.update(grads, state.opt_state, params)
    expected = state.replace(
        params=optax.apply_updates(params, updates), opt_state=expected_opt
    )
    mask = None
    if include_configuration:
        full_state = gin_model._merge_state_dicts(bundle['config'], params)
        state = jax_backend.TrainState(
            full_state, optimizer.init(full_state), full_state
        )
        graphdef = nnx.graphdef(model_builder._build_jax_model(predictor.model_config))
        mask = build_trainable_mask(SimpleNamespace(), full_state, module=graphdef)
    steps = jax_backend._build_train_functions(
        loss_fn,
        optimizer,
        grad_clip_value=None,
        use_ema=include_configuration,
        multi_device=True,
    )
    train_step = jax_backend._build_train_step(
        *steps,
        use_ema=include_configuration,
        ema_decay=0.9 if include_configuration else None,
        mask=mask,
    )
    state = jax_utils.replicate_global(state)
    actual, aux, _ = train_step(state, _device_batch([graph]), 0)
    if include_configuration:
        actual_params = flatten_dict(actual.params)
        for path, value in flatten_dict(expected.params).items():
            np.testing.assert_allclose(actual_params[path], value, rtol=3e-6, atol=2e-6)
        for path, value in flatten_dict(bundle['config']).items():
            _assert_tree_close(actual_params[path], value)
            _assert_tree_close(flatten_dict(actual.ema_params)[path], value)
        for before, after in zip(
            jax.tree_util.tree_leaves(state),
            jax.tree_util.tree_leaves(actual),
            strict=True,
        ):
            assert before.dtype == after.dtype
    else:
        _assert_tree_close(actual, expected)
    np.testing.assert_allclose(aux['metrics']['total'][0], expected_loss, rtol=2e-6)


def test_training_across_two_processes(tmp_path):
    """Use fresh CPU workers even when the main pytest process uses GPUs."""
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        address = f'127.0.0.1:{sock.getsockname()[1]}'
    env = {k: v for k, v in os.environ.items() if not k.lower().endswith('_proxy')}
    env.update(
        JAX_PLATFORMS='cpu',
        JAX_PLATFORM_NAME='cpu',
        JAX_NUM_CPU_DEVICES='1',
        CUDA_VISIBLE_DEVICES='',
        JAX_CPU_COLLECTIVES_IMPLEMENTATION='gloo',
        XLA_FLAGS='--xla_force_host_platform_device_count=1',
        OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',
        TMPDIR=str(tmp_path),
    )
    processes, logs = [], []
    timed_out = False
    try:
        for rank in range(2):
            log = (tmp_path / f'worker-{rank}.log').open('w')
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        '-m',
                        'tests.test_jax_distributed_training',
                        str(rank),
                        address,
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        deadline = time.monotonic() + 60
        for process in processes:
            process.wait(timeout=max(1, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
    output = '\n'.join(
        (tmp_path / f'worker-{rank}.log').read_text() for rank in range(2)
    )
    if (
        all(p.returncode == 77 for p in processes)
        and 'GLOO_PERMISSION_DENIED' in output
    ):
        pytest.skip('CPU distributed backend cannot access Gloo system information')
    assert not timed_out, f'Distributed workers timed out:\n{output}'
    assert all(p.returncode == 0 for p in processes), output
    assert output.count('DISTRIBUTED_CHECKS_PASSED') == 2, output


def _distributed_worker(rank, address):
    jax.distributed.initialize(
        coordinator_address=address,
        num_processes=2,
        process_id=rank,
        initialization_timeout=20,
    )
    try:
        jax.devices()
    except RuntimeError as exc:
        if 'gloo/common/linux.cc' in str(exc) and 'Permission denied' in str(exc):
            print(f'GLOO_PERMISSION_DENIED: {exc}', flush=True)
            sys.exit(77)
        raise
    assert jax.device_count() == 2

    def predict(params, graph):
        return {'energy': jnp.full(graph.n_node.shape, params['w'])}

    loss_fn = build_loss_fn(
        predict,
        LossSettings(
            energy_weight=1.0,
            forces_weight=0.0,
            stress_weight=0.0,
            loss_type_energy='mse',
            loss_energy_per_atom=False,
        ),
    )
    optimizer = optax.sgd(0.1)
    steps = jax_backend._build_train_functions(
        loss_fn, optimizer, grad_clip_value=None, use_ema=False, multi_device=True
    )
    train_step = jax_backend._build_train_step(*steps, use_ema=False, ema_decay=None)
    eval_step = jax_backend._build_eval_step(loss_fn, multi_device=True)
    for initially_empty in (False, True):
        # Start with different committed local states to verify broadcasting.
        params = jax.device_put(
            {'w': np.float32(7 * rank), 'unused': np.full((2, 3), rank, np.float32)},
            jax.local_devices()[0],
        )
        state = jax_utils.replicate_global(
            jax_backend.TrainState(params, optimizer.init(params), None), broadcast=True
        )
        assert float(state.params['w']) == 0
        np.testing.assert_array_equal(
            jax_backend._unreplicate(state.params['unused']), np.zeros((2, 3))
        )
        targets = (() if initially_empty else (1.0,)) if rank == 0 else (3.0, 5.0)
        loader = [_batch([_graph(2, t)]) for t in targets]
        final, metrics, _ = jax_backend._run_train_epoch(
            state,
            loader,
            train_step,
            max_steps=None,
            multi_device=True,
            ema_count_start=0,
            logger=None,
            args=SimpleNamespace(tqdm=False, verbose=0),
            epoch=1,
            metric_settings=dict(
                include_energy=True,
                include_forces=False,
                include_stress=False,
                loss_label='mse',
            ),
            learning_rate=0.1,
            is_primary=rank == 0,
        )
        np.testing.assert_allclose(
            final.params['w'], 1.48 if initially_empty else 1.32, rtol=1e-6
        )
        expected_loss = (9 + 4.4**2) / 2 if initially_empty else (10 + 4.6**2) / 3
        assert metrics.mean() == pytest.approx(expected_loss, rel=1e-6)
        assert final.params['unused'].shape == (2, 3)
        # Evaluation also accepts local committed parameters.
        eval_params = jax.device_put(
            {'w': np.float32(0), 'unused': np.zeros((2, 3), np.float32)},
            jax.local_devices()[0],
        )
        _, evaluated = jax_backend._run_eval_loop(
            eval_params, loader, eval_step, max_steps=None, multi_device=True
        )
        assert evaluated.mean() == pytest.approx(
            17 if initially_empty else 35 / 3, rel=1e-6
        )
    print('DISTRIBUTED_CHECKS_PASSED', flush=True)
    jax.distributed.shutdown()


if __name__ == '__main__':
    _distributed_worker(int(sys.argv[1]), sys.argv[2])
