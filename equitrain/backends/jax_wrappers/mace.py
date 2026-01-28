"""
JAX MACE wrapper.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from equitrain.backends.jax_runtime import ensure_multiprocessing_spawn
from equitrain.data.atomic import AtomicNumberTable

ensure_multiprocessing_spawn()


class MaceWrapper:
    """
    Thin wrapper around a MACE-JAX module mirroring the torch counterpart.
    """

    def __init__(
        self,
        module: Any,
        config: dict[str, Any],
        *,
        compute_force: bool = False,
        compute_stress: bool = False,
    ):
        self.module = module
        self.config = dict(config)
        self.compute_force = compute_force
        self.compute_stress = compute_stress

    def apply(
        self,
        variables: dict[str, Any],
        data_dict: dict[str, jnp.ndarray],
        *,
        compute_force: bool | None = None,
        compute_stress: bool | None = None,
    ) -> dict[str, jnp.ndarray]:
        force_flag = self.compute_force if compute_force is None else compute_force
        stress_flag = self.compute_stress if compute_stress is None else compute_stress

        if hasattr(self.module, 'init'):
            return self.module.apply(
                variables,
                data_dict,
                compute_force=force_flag,
                compute_stress=stress_flag,
            )

        # NNX graphdef path (module.apply returns a callable).
        outputs, _ = self.module.apply(variables)(
            data_dict,
            compute_force=force_flag,
            compute_stress=stress_flag,
        )
        return outputs

    @property
    def atomic_numbers(self) -> AtomicNumberTable | None:
        numbers = self.config.get('atomic_numbers')
        if numbers is None:
            return None
        return AtomicNumberTable([int(z) for z in numbers])

    @property
    def atomic_energies(self):
        atomic_energies = self.config.get('atomic_energies')
        if atomic_energies is None:
            return None
        if isinstance(atomic_energies, dict):
            numbers = self.config.get('atomic_numbers', [])
            return [
                atomic_energies.get(str(int(z)))
                if str(int(z)) in atomic_energies
                else atomic_energies.get(int(z))
                for z in numbers
            ]
        if isinstance(atomic_energies, list | tuple | np.ndarray | jnp.ndarray):
            return list(np.asarray(atomic_energies, dtype=float))
        return atomic_energies

    @property
    def r_max(self) -> float | None:
        value = self.config.get('r_max')
        return float(value) if value is not None else None

    @r_max.setter
    def r_max(self, value: float):
        self.config['r_max'] = float(value)
        try:
            from flax import nnx
            from mace_jax.cli import mace_jax_from_torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'mace_jax is required to modify the cutoff radius for JAX models.'
            ) from exc
        module = mace_jax_from_torch._build_jax_model(self.config, rngs=nnx.Rngs(0))
        if hasattr(module, 'init'):
            self.module = module
        else:
            graphdef, _ = nnx.split(module)
            self.module = graphdef

    def with_compute_flags(
        self, *, force: bool | None = None, stress: bool | None = None
    ):
        if force is not None:
            self.compute_force = force
        if stress is not None:
            self.compute_stress = stress
        return self


def build_module(config: dict[str, Any]):
    """
    Build a JAX MACE module and its template data from a configuration dict.

    This helper lives alongside the wrapper so that the backend can stay agnostic
    of the model-specific dependencies (``mace_jax`` in this case).
    """

    try:
        from mace_jax.cli import mace_jax_from_torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            'mace_jax is required to load MACE models for the JAX backend.'
        ) from exc

    from flax import nnx

    module = mace_jax_from_torch._build_jax_model(config, rngs=nnx.Rngs(0))
    template = mace_jax_from_torch._prepare_template_data(config)
    return module, template


__all__ = ['MaceWrapper', 'build_module']
