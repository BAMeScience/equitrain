from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('jax', reason='JAX is required for jax calculator tests.')
pytest.importorskip('flax', reason='Flax is required for jax calculator tests.')
pytest.importorskip('jraph', reason='jraph is required for jax calculator tests.')
pytest.importorskip('ase', reason='ASE is required for calculator tests.')

import jax.numpy as jnp  # noqa: E402
from ase import Atoms  # noqa: E402

import equitrain.calculators.jax_wrapper as jax_calculators  # noqa: E402
from equitrain.backends.jax_utils import ModelBundle  # noqa: E402


class _DummyAniModule:
    def apply(self, variables, inputs):
        del variables
        coords = inputs['coordinates']
        energy = jnp.sum(coords**2, axis=(1, 2))
        forces = -2.0 * coords
        return {'energy': energy, 'forces': forces}


def _build_dummy_bundle() -> ModelBundle:
    return ModelBundle(
        config={
            'atomic_numbers': [1, 8],
            'r_max': 3.0,
            'wrapper_name': 'ani',
        },
        params={},
        module=_DummyAniModule(),
    )


def test_jax_wrapper_predictor_predict():
    predictor = jax_calculators.JaxWrapperPredictor(
        model=_build_dummy_bundle(),
        model_wrapper='ani',
        device='cpu',
        batch_size=8,
        require_forces=True,
    )
    atoms = [
        Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        Atoms(numbers=[8, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
    ]
    energies, forces = predictor.predict(atoms, require_forces=True)

    np.testing.assert_allclose(np.asarray(energies), np.array([1.0, 4.0], dtype=float))
    assert forces is not None
    assert len(forces) == 2
    assert forces[0].shape == (2, 3)
    assert forces[1].shape == (2, 3)


def test_build_jax_ase_calculator():
    calc = jax_calculators.build_jax_ase_calculator(
        model=_build_dummy_bundle(),
        model_wrapper='ani',
        device='cpu',
        batch_size=4,
    )
    atoms = Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert isinstance(float(energy), float)
    assert forces.shape == (2, 3)
