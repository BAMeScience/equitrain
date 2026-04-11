from __future__ import annotations

import json

import pytest

pytest.importorskip('jax', reason='JAX runtime is required for JAX backend tests.')
pytest.importorskip('flax', reason='Flax is required for JAX backend tests.')

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import serialization  # noqa: E402

from equitrain.backends import jax_utils  # noqa: E402
from equitrain.backends.jax_wrappers.ani import AniWrapper  # noqa: E402


class _CaptureSpeciesModule:
    def __init__(self):
        self.captured_species = None

    def apply(self, variables, species, coordinates):
        del variables, coordinates
        self.captured_species = np.asarray(species)
        return {'energy': jnp.zeros((species.shape[0],), dtype=jnp.float32)}


class _EnergyOnlyModule:
    def apply(self, variables, species, coordinates):
        del variables, species
        energy = jnp.sum(coordinates**2, axis=(1, 2))
        return {'energy': energy}


class _MaskedEnergyModule:
    def apply(self, variables, inputs):
        del variables
        coordinates = inputs['coordinates']
        atom_mask = inputs['atom_mask']
        energy = jnp.sum(
            jnp.where(atom_mask[..., None], coordinates**2, 0.0),
            axis=(1, 2),
        )
        return {'energy': energy}


class _VoigtStressModule:
    def apply(self, variables, inputs):
        del variables
        energy = jnp.zeros((inputs['species'].shape[0],), dtype=jnp.float32)
        stress = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=jnp.float32)
        return {'energy': energy, 'stress': stress}


def test_jax_ani_wrapper_remaps_dataset_species_to_model_order():
    module = _CaptureSpeciesModule()
    wrapper = AniWrapper(
        module=module,
        config={
            'atomic_numbers': [1, 6],
            'species_order': ['C', 'H'],
        },
    )

    wrapper.apply(
        {},
        {
            'node_attrs_index': jnp.array([0, 1], dtype=jnp.int32),
            'positions': jnp.zeros((2, 3), dtype=jnp.float32),
            'ptr': jnp.array([0, 2], dtype=jnp.int32),
        },
    )

    np.testing.assert_array_equal(
        module.captured_species,
        np.array([[1, 0]], dtype=np.int32),
    )


def test_jax_ani_wrapper_supports_direct_species_coordinate_inputs():
    wrapper = AniWrapper(
        module=_EnergyOnlyModule(),
        config={'atomic_numbers': [1, 6], 'r_max': 5.2},
        compute_force=True,
    )

    outputs = wrapper.apply(
        {},
        jnp.array([[0, 1, -1]], dtype=jnp.int32),
        jnp.array(
            [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=jnp.float32,
        ),
    )

    np.testing.assert_allclose(outputs['energy'], np.array([5.0], dtype=np.float32))
    np.testing.assert_allclose(
        outputs['forces'],
        np.array([[-2.0, 0.0, 0.0], [0.0, -4.0, 0.0]], dtype=np.float32),
    )


def test_jax_ani_wrapper_direct_inputs_are_already_model_species():
    module = _CaptureSpeciesModule()
    wrapper = AniWrapper(
        module=module,
        config={
            'atomic_numbers': [1, 6],
            'species_order': ['C', 'H'],
        },
    )

    wrapper.apply(
        {},
        jnp.array([[0, 1]], dtype=jnp.int32),
        jnp.zeros((1, 2, 3), dtype=jnp.float32),
    )

    np.testing.assert_array_equal(
        module.captured_species,
        np.array([[0, 1]], dtype=np.int32),
    )


def test_jax_ani_wrapper_computes_forces_when_module_only_returns_energy():
    wrapper = AniWrapper(
        module=_EnergyOnlyModule(),
        config={'atomic_numbers': [1, 6], 'r_max': 5.2},
        compute_force=True,
        compute_stress=True,
    )

    outputs = wrapper.apply(
        {},
        {
            'node_attrs_index': jnp.array([0, 1], dtype=jnp.int32),
            'positions': jnp.array(
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=jnp.float32
            ),
            'ptr': jnp.array([0, 2], dtype=jnp.int32),
        },
    )

    np.testing.assert_allclose(outputs['energy'], np.array([5.0], dtype=np.float32))
    np.testing.assert_allclose(
        outputs['forces'],
        np.array([[-2.0, 0.0, 0.0], [0.0, -4.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        outputs['stress'],
        np.zeros((1, 3, 3), dtype=np.float32),
    )


def test_jax_ani_wrapper_ignores_padding_and_jits():
    wrapper = AniWrapper(
        module=_MaskedEnergyModule(),
        config={'atomic_numbers': [1, 6], 'r_max': 5.2},
        compute_force=True,
        compute_stress=True,
    )
    data = {
        'node_attrs_index': jnp.array([0, 1, 0, 0], dtype=jnp.int32),
        'node_attrs': jnp.array(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=jnp.float32,
        ),
        'positions': jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
            dtype=jnp.float32,
        ),
        'ptr': jnp.array([0, 2, 4], dtype=jnp.int32),
    }

    outputs = jax.jit(lambda inputs: wrapper.apply({}, inputs))(data)

    np.testing.assert_allclose(
        outputs['energy'], np.array([5.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        outputs['forces'],
        np.array(
            [
                [-2.0, 0.0, 0.0],
                [0.0, -4.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        outputs['stress'],
        np.zeros((2, 3, 3), dtype=np.float32),
    )


def test_jax_ani_wrapper_converts_voigt_stress_to_full_matrix():
    wrapper = AniWrapper(
        module=_VoigtStressModule(),
        config={'atomic_numbers': [1, 6], 'r_max': 5.2},
    )

    outputs = wrapper.apply(
        {},
        {
            'node_attrs_index': jnp.array([0, 1], dtype=jnp.int32),
            'node_attrs': jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
            'positions': jnp.zeros((2, 3), dtype=jnp.float32),
            'ptr': jnp.array([0, 2], dtype=jnp.int32),
        },
    )

    np.testing.assert_allclose(
        outputs['stress'],
        np.array(
            [[[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]]],
            dtype=np.float32,
        ),
    )


def test_load_model_bundle_supports_non_mace_wrappers_without_mace_jax(
    tmp_path, monkeypatch
):
    config = {
        'wrapper_name': 'ani',
        'atomic_numbers': [1, 6],
        'r_max': 5.2,
    }
    model_dir = tmp_path / 'ani_bundle'
    model_dir.mkdir()
    (model_dir / 'config.json').write_text(json.dumps(config))

    params_template = {'params': {'scale': jnp.array([2.0], dtype=jnp.float32)}}
    (model_dir / 'params.msgpack').write_bytes(serialization.to_bytes(params_template))

    def fake_builder(_config):
        return object(), params_template

    monkeypatch.setattr(jax_utils, 'get_wrapper_builder', lambda name: fake_builder)

    bundle = jax_utils.load_model_bundle(str(model_dir), dtype='float32', wrapper='ani')

    assert bundle.config['wrapper_name'] == 'ani'
    np.testing.assert_allclose(
        bundle.params['params']['scale'], np.array([2.0], dtype=np.float32)
    )
