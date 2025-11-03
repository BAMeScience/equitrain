"""
JAX-specific wrappers that provide a higher level interface similar to the torch
wrappers. They encapsulate a Flax/Equinox style model (module + parameters) and
expose helpers for metadata and applying the model with sensible defaults.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from equitrain.backends.jax_runtime import ensure_multiprocessing_spawn
from equitrain.data.atomic import AtomicNumberTable

ensure_multiprocessing_spawn()


class MaceWrapper:
    """
    Thin wrapper around a MACE-JAX module that mirrors the torch wrapper
    interface. It stores metadata extracted from the configuration so callers
    can query atomic numbers / energies / cutoffs in a uniform way.
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
        """
        Run a forward pass of the wrapped module.

        Parameters
        ----------
        variables:
            PyTree of parameters/state for the module.
        data_dict:
            Dictionary of arrays matching the input structure expected by the
            JAX MACE model.
        compute_force, compute_stress:
            Optional overrides for the default behaviour.
        """
        return self.module.apply(
            variables,
            data_dict,
            compute_force=self.compute_force
            if compute_force is None
            else compute_force,
            compute_stress=self.compute_stress
            if compute_stress is None
            else compute_stress,
        )

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
        """
        Update the cutoff radius. This rebuilds the underlying module to keep
        the configuration in sync.
        """
        self.config['r_max'] = float(value)
        try:
            from mace_jax.cli import mace_torch2jax
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                'mace_jax is required to modify the cutoff radius for JAX models.'
            ) from exc
        self.module = mace_torch2jax._build_jax_model(self.config)

    def with_compute_flags(
        self, *, force: bool | None = None, stress: bool | None = None
    ):
        if force is not None:
            self.compute_force = force
        if stress is not None:
            self.compute_stress = stress
        return self


__all__ = ['MaceWrapper']
