from __future__ import annotations

from types import SimpleNamespace

import pytest

from equitrain import get_args_parser_evaluate
from equitrain.backends import jax_evaluate
from equitrain.backends.jax_utils import ModelBundle
from equitrain.backends.jax_loss import JaxLossCollection


class _DummyLoader:
    def __init__(self, items):
        self._items = list(items)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


def _default_args() -> SimpleNamespace:
    args = get_args_parser_evaluate().parse_args([])
    args.model = 'dummy-model'
    args.test_file = 'dummy-data'
    args.batch_size = 4
    args.batch_max_nodes = None
    args.batch_max_edges = None
    args.forces_weight = 0.0
    args.stress_weight = 0.0
    return args


def test_jax_evaluate_multi_device_path(monkeypatch):
    captured: dict[str, object] = {}

    class DummyLogger:
        def log(self, *args, **kwargs):
            captured.setdefault('log_calls', []).append((args, kwargs))

    bundle = ModelBundle(
        config={'atomic_numbers': [1], 'r_max': 3.0},
        params={'weights': 1.0},
        module=object(),
    )

    def fake_build_eval_loss(apply_fn, settings):
        captured['build_eval_loss_called'] = True

        def loss_fn(params, graph):
            captured.setdefault('loss_fn_calls', []).append((params, graph))
            return 0.0, {'dummy': 0.0}

        return loss_fn

    def fake_build_loader(graphs, **_):
        return _DummyLoader(graphs)

    def fake_atoms_to_graphs(path, *_):
        captured['atoms_to_graphs_path'] = path
        return ['g0', 'g1', 'g2', 'g3']

    def fake_make_apply_fn(wrapper, num_species):
        captured['apply_fn_species'] = num_species

        def apply_fn(params, graph):
            captured.setdefault('apply_calls', []).append((params, graph))
            return {'energy': 0.0}

        return apply_fn

    def fake_jax_wrapper(*args, **kwargs):
        captured['wrapper_kwargs'] = kwargs
        return SimpleNamespace(compute_force=False, compute_stress=False)

    def fake_build_eval_step(loss_fn, *, multi_device):
        captured['multi_device_in_eval_step'] = multi_device

        def step(params, batch):
            captured.setdefault('eval_step_calls', []).append((params, batch))
            return loss_fn(params, batch)

        return step

    def fake_run_eval_loop(params, loader, eval_step_fn, *, max_steps, multi_device):
        captured['run_eval_loop_multi_device'] = multi_device
        captured['run_eval_loop_max_steps'] = max_steps
        captured['run_eval_loop_params'] = params
        captured['run_eval_loop_loader'] = list(loader)
        return 0.5, JaxLossCollection()

    def fake_device_put_replicated(params, devices):
        captured['device_put_replicated_devices'] = tuple(devices)
        return tuple(params for _ in devices)

    args = _default_args()

    monkeypatch.setattr(jax_evaluate, 'validate_evaluate_args', lambda *a, **k: None)
    monkeypatch.setattr(jax_evaluate, 'init_logger', lambda *a, **k: DummyLogger())
    monkeypatch.setattr(jax_evaluate, 'load_model_bundle', lambda *a, **k: bundle)
    monkeypatch.setattr(jax_evaluate, 'atoms_to_graphs', fake_atoms_to_graphs)
    monkeypatch.setattr(jax_evaluate, 'build_loader', fake_build_loader)
    monkeypatch.setattr(jax_evaluate, 'make_apply_fn', fake_make_apply_fn)
    monkeypatch.setattr(jax_evaluate, 'JaxMaceWrapper', fake_jax_wrapper)
    monkeypatch.setattr(jax_evaluate, 'build_eval_loss', fake_build_eval_loss)
    monkeypatch.setattr(jax_evaluate, '_build_eval_step', fake_build_eval_step)
    monkeypatch.setattr(jax_evaluate, '_run_eval_loop', fake_run_eval_loop)
    monkeypatch.setattr(jax_evaluate, '_is_multi_device', lambda: True)
    monkeypatch.setattr(jax_evaluate.jax, 'local_device_count', lambda: 2)
    monkeypatch.setattr(jax_evaluate.jax, 'local_devices', lambda: ('d0', 'd1'))
    monkeypatch.setattr(
        jax_evaluate.jax, 'device_put_replicated', fake_device_put_replicated
    )

    result = jax_evaluate.evaluate(args)

    assert result == 0.0
    assert captured['multi_device_in_eval_step'] is True
    assert captured['run_eval_loop_multi_device'] is True
    assert captured['run_eval_loop_loader'] == ['g0', 'g1', 'g2', 'g3']
    assert captured['run_eval_loop_params'] == ({'weights': 1.0}, {'weights': 1.0})
    assert captured['device_put_replicated_devices'] == ('d0', 'd1')
    assert captured['wrapper_kwargs']['compute_force'] is False
    assert captured['wrapper_kwargs']['compute_stress'] is False


def test_jax_evaluate_batch_size_check(monkeypatch):
    args = _default_args()
    args.batch_size = 3

    bundle = ModelBundle(
        config={'atomic_numbers': [1], 'r_max': 3.0},
        params={'weights': 1.0},
        module=object(),
    )

    monkeypatch.setattr(jax_evaluate, 'validate_evaluate_args', lambda *a, **k: None)
    monkeypatch.setattr(jax_evaluate, 'load_model_bundle', lambda *a, **k: bundle)
    monkeypatch.setattr(jax_evaluate, 'atoms_to_graphs', lambda *a, **k: ['g0'])
    monkeypatch.setattr(jax_evaluate, '_is_multi_device', lambda: True)
    monkeypatch.setattr(jax_evaluate.jax, 'local_device_count', lambda: 2)

    with pytest.raises(ValueError, match='--batch-size must be divisible'):
        jax_evaluate.evaluate(args)
