from types import SimpleNamespace

import numpy as np
import torch.serialization

torch.serialization.add_safe_globals([slice])

from equitrain import get_args_parser_predict
from equitrain.backends import jax_predict


def test_jax_predict_basic(monkeypatch):
    records = {'graphs': []}

    def fake_load_model_bundle(path, dtype=None):
        records['bundle_path'] = path
        return SimpleNamespace(
            params={'weights': 1.0},
            module=None,
            config={'atomic_numbers': [1], 'r_max': 2.5},
        )

    monkeypatch.setattr(jax_predict, '_load_bundle', fake_load_model_bundle)

    def fake_atoms_to_graphs(path, r_max, z_table, **kwargs):
        records['atoms_args'] = (path, r_max, z_table, kwargs)
        return ['g1', 'g2']

    monkeypatch.setattr(jax_predict, 'atoms_to_graphs', fake_atoms_to_graphs)
    monkeypatch.setattr(jax_predict, 'build_loader', lambda graphs, **_: graphs)
    monkeypatch.setattr(jax_predict, '_prepare_single_batch', lambda graph: graph)
    monkeypatch.setattr(jax_predict, '_is_multi_device', lambda: False)

    class DummyWrapper:
        def __init__(self):
            self.compute_force = True
            self.compute_stress = False

    monkeypatch.setattr(
        jax_predict, '_create_wrapper', lambda *args, **kwargs: DummyWrapper()
    )

    def fake_make_apply_fn(wrapper, num_species):
        def _impl(params, batch):
            records.setdefault('apply_calls', []).append((params, batch))
            return {
                'energy': np.array([1.0]),
                'forces': np.array([[0.0, 0.0, 0.0]]),
                'stress': None,
            }

        return _impl

    monkeypatch.setattr(jax_predict, 'make_apply_fn', fake_make_apply_fn)

    monkeypatch.setattr(jax_predict.jax, 'jit', lambda fn: fn)
    monkeypatch.setattr(jax_predict.jax, 'device_get', lambda x: x)

    args = get_args_parser_predict().parse_args([])
    args.backend = 'jax'
    args.predict_file = 'predict.h5'
    args.model = 'model.bundle'
    args.batch_size = 1
    args.forces_weight = 1.0
    args.stress_weight = 0.0
    args.dtype = 'float32'
    args.niggli_reduce = True

    energy, forces, stress = jax_predict.predict(args)

    assert energy.shape == (2,)
    assert forces.shape[0] == 2
    assert stress is None
    assert records['bundle_path'] == 'model.bundle'
    assert records['atoms_args'][3]['niggli_reduce'] is True
