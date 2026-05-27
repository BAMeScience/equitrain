from __future__ import annotations

import json

import pytest

pytest.importorskip('jax', reason='JAX runtime is required for JAX backend tests.')
pytest.importorskip('flax', reason='Flax is required for JAX backend tests.')

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from equitrain.backends import jax_utils
from equitrain.backends.jax_wrappers.m3gnet import M3GNetWrapper


class _EnergyOnlyGraphModule:
    def __init__(self):
        self.captured_keys = None

    def apply(self, variables, inputs):
        del variables
        self.captured_keys = set(inputs)
        positions = inputs['positions']
        node_mask = inputs['node_mask']
        batch = inputs['batch']
        n_graphs = inputs['ptr'].shape[0] - 1
        per_node = jnp.sum(
            jnp.where(node_mask[:, None], positions**2, 0.0),
            axis=1,
        )
        return {'energy': jax.ops.segment_sum(per_node, batch, n_graphs)}


class _DenseForceModule:
    def apply(self, variables, inputs):
        del variables
        energy = jnp.zeros((inputs['ptr'].shape[0] - 1,), dtype=jnp.float32)
        forces = jnp.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [9.0, 9.0, 9.0]],
                [[3.0, 0.0, 0.0], [9.0, 9.0, 9.0], [9.0, 9.0, 9.0]],
            ],
            dtype=jnp.float32,
        )
        return {'energy': energy, 'forces': forces}


def _data_dict():
    return {
        'positions': jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        ),
        'node_attrs_index': jnp.array([0, 1, 0], dtype=jnp.int32),
        'edge_index': jnp.array([[0, 1], [1, 0]], dtype=jnp.int32),
        'shifts': jnp.zeros((2, 3), dtype=jnp.float32),
        'unit_shifts': jnp.zeros((2, 3), dtype=jnp.float32),
        'batch': jnp.array([0, 0, 1], dtype=jnp.int32),
        'ptr': jnp.array([0, 2, 3], dtype=jnp.int32),
        'cell': jnp.tile(jnp.eye(3, dtype=jnp.float32)[None, :, :], (2, 1, 1)),
        'node_mask': jnp.array([True, True, False]),
    }


def test_jax_m3gnet_wrapper_computes_forces_and_aliases_graph_inputs():
    module = _EnergyOnlyGraphModule()
    wrapper = M3GNetWrapper(
        module=module,
        config={'atomic_numbers': [11, 17], 'r_max': 5.0},
        compute_force=True,
        compute_stress=True,
    )

    outputs = wrapper.apply({}, _data_dict())

    np.testing.assert_allclose(outputs['energy'], np.array([5.0, 0.0]))
    np.testing.assert_allclose(
        outputs['forces'],
        np.array(
            [
                [-2.0, 0.0, 0.0],
                [0.0, -4.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(outputs['stress'], np.zeros((2, 3, 3)))
    assert {'node_type', 'pos', 'pbc_offshift', 'pbc_offset'} <= module.captured_keys


def test_jax_m3gnet_wrapper_flattens_dense_forces():
    wrapper = M3GNetWrapper(
        module=_DenseForceModule(),
        config={'atomic_numbers': [11, 17], 'r_max': 5.0},
    )

    outputs = wrapper.apply({}, _data_dict())

    np.testing.assert_allclose(
        outputs['forces'],
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_load_model_bundle_supports_m3gnet_without_mace_jax(tmp_path, monkeypatch):
    config = {
        'wrapper_name': 'm3gnet',
        'atomic_numbers': [11, 17],
        'r_max': 5.0,
    }
    model_dir = tmp_path / 'm3gnet_bundle'
    model_dir.mkdir()
    (model_dir / 'config.json').write_text(json.dumps(config))

    params_template = {'params': {'scale': jnp.array([2.0], dtype=jnp.float32)}}
    (model_dir / 'params.msgpack').write_bytes(serialization.to_bytes(params_template))

    def fake_builder(_config):
        return object(), params_template

    monkeypatch.setattr(jax_utils, 'get_wrapper_builder', lambda name: fake_builder)

    bundle = jax_utils.load_model_bundle(
        str(model_dir),
        dtype='float32',
        wrapper='m3gnet',
    )

    assert bundle.config['wrapper_name'] == 'm3gnet'
    np.testing.assert_allclose(
        bundle.params['params']['scale'],
        np.array([2.0], dtype=np.float32),
    )


def test_m3gnet_build_module_accepts_factory_returning_template(monkeypatch):
    from equitrain.backends.jax_wrappers import m3gnet

    params_template = {'params': {'scale': jnp.array([1.0], dtype=jnp.float32)}}
    module = object()

    monkeypatch.setattr(
        m3gnet,
        '_import_symbol',
        lambda path: (lambda **kwargs: (module, params_template)),
    )

    built_module, built_template = m3gnet.build_module(
        {'module_factory': 'pkg:create_model'}
    )

    assert built_module is module
    assert built_template is params_template
