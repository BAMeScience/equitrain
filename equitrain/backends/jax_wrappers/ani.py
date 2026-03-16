"""
JAX ANI wrapper.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from equitrain.backends.jax_runtime import ensure_multiprocessing_spawn
from equitrain.data.atomic import AtomicNumberTable

ensure_multiprocessing_spawn()

_SYMBOL_TO_Z = {
    'H': 1,
    'He': 2,
    'Li': 3,
    'Be': 4,
    'B': 5,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'Ne': 10,
    'Na': 11,
    'Mg': 12,
    'Al': 13,
    'Si': 14,
    'P': 15,
    'S': 16,
    'Cl': 17,
    'Ar': 18,
}


def _import_symbol(path: str):
    module_name, _, attr_name = path.replace(':', '.').rpartition('.')
    if not module_name or not attr_name:
        raise ValueError(
            f'Invalid import path "{path}". Expected "package.module:attribute".'
        )
    module = import_module(module_name)
    return getattr(module, attr_name)


class AniWrapper:
    """
    Thin wrapper around a JAX-native ANI-style module.

    The wrapped module is expected to accept either a mapping with
    ``species``/``coordinates`` entries or the positional arguments
    ``(species, coordinates)`` and to return either:

    - a mapping containing ``energy`` and optionally ``forces`` / ``stress``
    - an object with an ``energies`` attribute
    - a tuple/list whose second item is the energy tensor
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

    def _prepare_batch_inputs(
        self,
        data_dict: dict[str, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        species = jnp.asarray(data_dict['node_attrs_index'], dtype=jnp.int32)
        coordinates = jnp.asarray(data_dict['positions'])
        ptr = jnp.asarray(data_dict['ptr'], dtype=jnp.int32)

        counts = ptr[1:] - ptr[:-1]
        batch_size = counts.shape[0]
        max_atoms = jnp.max(
            jnp.concatenate([counts, jnp.zeros((1,), dtype=counts.dtype)])
        )

        graph_index = jnp.arange(batch_size, dtype=jnp.int32)
        batch = jnp.repeat(graph_index, counts, total_repeat_length=species.shape[0])
        local_index = jnp.arange(species.shape[0], dtype=jnp.int32) - ptr[batch]

        species_batch = jnp.full((batch_size, max_atoms), -1, dtype=jnp.int32)
        coordinates_batch = jnp.zeros(
            (batch_size, max_atoms, 3),
            dtype=coordinates.dtype,
        )
        atom_mask = jnp.zeros((batch_size, max_atoms), dtype=bool)

        species_batch = species_batch.at[batch, local_index].set(species)
        coordinates_batch = coordinates_batch.at[batch, local_index].set(coordinates)
        atom_mask = atom_mask.at[batch, local_index].set(True)

        return species_batch, coordinates_batch, atom_mask, counts

    def _call_module(
        self,
        variables: dict[str, Any],
        species_batch: jnp.ndarray,
        coordinates_batch: jnp.ndarray,
        atom_mask: jnp.ndarray,
        counts: jnp.ndarray,
    ):
        model_inputs = {
            'species': species_batch,
            'coordinates': coordinates_batch,
            'atom_mask': atom_mask,
            'counts': counts,
        }

        if hasattr(self.module, 'init'):
            try:
                return self.module.apply(variables, model_inputs)
            except TypeError:
                return self.module.apply(variables, species_batch, coordinates_batch)

        apply_fn = self.module.apply(variables)
        try:
            outputs, _ = apply_fn(model_inputs)
        except TypeError:
            outputs, _ = apply_fn(species_batch, coordinates_batch)
        return outputs

    def _normalize_outputs(
        self,
        outputs,
        *,
        counts: jnp.ndarray,
        atom_mask: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        if isinstance(outputs, dict):
            result = dict(outputs)
        elif hasattr(outputs, 'energies'):
            result = {'energy': getattr(outputs, 'energies')}
            if hasattr(outputs, 'forces'):
                result['forces'] = getattr(outputs, 'forces')
            if hasattr(outputs, 'stress'):
                result['stress'] = getattr(outputs, 'stress')
        elif isinstance(outputs, list | tuple):
            if len(outputs) < 2:
                raise ValueError('ANI module returned a tuple/list without energies.')
            result = {'energy': outputs[1]}
        else:
            result = {'energy': outputs}

        energy = result.get('energy', result.get('energies'))
        if energy is None:
            raise ValueError('ANI module output must contain energies.')
        energy = jnp.asarray(energy).reshape(-1)
        result['energy'] = energy[: counts.shape[0]]

        if 'forces' in result and result['forces'] is not None:
            forces = jnp.asarray(result['forces'])
            if forces.ndim == 3:
                result['forces'] = forces[atom_mask]
            else:
                result['forces'] = forces

        if 'stress' in result and result['stress'] is not None:
            stress = jnp.asarray(result['stress'])
            if stress.ndim == 2 and stress.shape[-1] == 6:
                stress = stress.reshape(stress.shape[0], 6)
            result['stress'] = stress[: counts.shape[0]]

        return result

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

        species_batch, coordinates_batch, atom_mask, counts = self._prepare_batch_inputs(
            data_dict
        )

        def _forward(coords):
            outputs = self._call_module(
                variables,
                species_batch,
                coords,
                atom_mask,
                counts,
            )
            return self._normalize_outputs(
                outputs,
                counts=counts,
                atom_mask=atom_mask,
            )

        normalized = _forward(coordinates_batch)

        result = {
            'energy': normalized['energy'],
            'forces': normalized.get('forces'),
            'stress': normalized.get('stress'),
        }

        if force_flag and result['forces'] is None:
            def _total_energy(coords):
                return jnp.sum(_forward(coords)['energy'])

            forces_full = -jax.grad(_total_energy)(coordinates_batch)
            result['forces'] = forces_full[atom_mask]

        if stress_flag and result['stress'] is None:
            result['stress'] = jnp.zeros(
                (counts.shape[0], 3, 3),
                dtype=coordinates_batch.dtype,
            )

        return result

    @property
    def atomic_numbers(self) -> AtomicNumberTable | None:
        numbers = self.config.get('atomic_numbers')
        if numbers is not None:
            return AtomicNumberTable([int(z) for z in numbers])

        species_order = self.config.get('species_order')
        if species_order is None:
            return AtomicNumberTable([1, 6, 7, 8])

        resolved = sorted(
            {
                _SYMBOL_TO_Z.get(str(symbol), symbol)
                for symbol in species_order
            }
        )
        return AtomicNumberTable([int(z) for z in resolved])

    @property
    def atomic_energies(self):
        atomic_energies = self.config.get('atomic_energies')
        if atomic_energies is None:
            return None
        if isinstance(atomic_energies, dict):
            numbers = list(self.atomic_numbers or [])
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
    def r_max(self) -> float:
        return float(self.config.get('r_max', 5.2))

    @r_max.setter
    def r_max(self, value: float):
        self.config['r_max'] = float(value)

    def with_compute_flags(
        self, *, force: bool | None = None, stress: bool | None = None
    ):
        if force is not None:
            self.compute_force = force
        if stress is not None:
            self.compute_stress = stress
        return self


def build_module(config: dict[str, Any]):
    factory_path = config.get('module_factory') or config.get('module_builder')
    class_path = config.get('module_class')
    kwargs = dict(config.get('model_kwargs', {}))

    if factory_path:
        factory = _import_symbol(str(factory_path))
        module = factory(**kwargs)
    elif class_path:
        module_class = _import_symbol(str(class_path))
        module = module_class(**kwargs)
    else:
        raise ValueError(
            'ANI JAX config must define `module_factory`, `module_builder`, or `module_class`.'
        )

    return module, None


__all__ = ['AniWrapper', 'build_module']
