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


def _voigt6_to_full3x3(stress: jnp.ndarray) -> jnp.ndarray:
    sxx, syy, szz, syz, sxz, sxy = [stress[..., idx] for idx in range(6)]
    return jnp.stack(
        [
            jnp.stack([sxx, sxy, sxz], axis=-1),
            jnp.stack([sxy, syy, syz], axis=-1),
            jnp.stack([sxz, syz, szz], axis=-1),
        ],
        axis=-2,
    )


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

    def _species_order_numbers(self) -> list[int] | None:
        species_order = self.config.get('species_order')
        if species_order is None:
            return None

        resolved = []
        for symbol in species_order:
            number = _SYMBOL_TO_Z.get(str(symbol), symbol)
            resolved.append(int(number))
        return resolved

    def _species_index_remap(self) -> jnp.ndarray | None:
        dataset_numbers = self.config.get('atomic_numbers')
        model_numbers = self._species_order_numbers()

        if dataset_numbers is None or model_numbers is None:
            return None

        index_by_number = {int(z): idx for idx, z in enumerate(model_numbers)}
        remap = []
        for number in dataset_numbers:
            z = int(number)
            if z not in index_by_number:
                raise ValueError(
                    'ANI species_order does not cover every atomic number in '
                    f'config.atomic_numbers: missing Z={z}.'
                )
            remap.append(index_by_number[z])
        return jnp.asarray(remap, dtype=jnp.int32)

    def _remap_species(self, species: jnp.ndarray) -> jnp.ndarray:
        remap = self._species_index_remap()
        if remap is None:
            return species

        valid = species >= 0
        safe_species = jnp.where(valid, species, 0)
        remapped = remap[safe_species]
        return jnp.where(valid, remapped, -1)

    def _node_mask_from_data(self, data_dict: dict[str, jnp.ndarray]) -> jnp.ndarray:
        if 'node_mask' in data_dict:
            return jnp.asarray(data_dict['node_mask'], dtype=bool)
        if 'node_attrs' in data_dict:
            node_attrs = jnp.asarray(data_dict['node_attrs'])
            return jnp.sum(jnp.abs(node_attrs), axis=-1) > 0
        return jnp.ones((jnp.asarray(data_dict['positions']).shape[0],), dtype=bool)

    def _prepare_direct_inputs(
        self,
        species: jnp.ndarray,
        coordinates: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict | None]:
        species_batch = jnp.asarray(species, dtype=jnp.int32)
        coordinates_batch = jnp.asarray(coordinates)

        if species_batch.ndim == 1:
            species_batch = species_batch[None, :]
        if coordinates_batch.ndim == 2:
            coordinates_batch = coordinates_batch[None, :, :]
        if species_batch.ndim != 2 or coordinates_batch.ndim != 3:
            raise ValueError(
                'ANI direct inputs must have species shape [batch, atoms] and '
                'coordinates shape [batch, atoms, 3].'
            )
        if coordinates_batch.shape[-1] != 3:
            raise ValueError('ANI coordinates must have trailing dimension 3.')

        atom_mask = species_batch >= 0
        counts = jnp.sum(atom_mask, axis=1, dtype=jnp.int32)
        return species_batch, coordinates_batch, atom_mask, counts, None

    def _prepare_batch_inputs(
        self,
        data_dict: dict[str, jnp.ndarray],
    ) -> tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]
    ]:
        species = jnp.asarray(data_dict['node_attrs_index'], dtype=jnp.int32)
        coordinates = jnp.asarray(data_dict['positions'])
        ptr = jnp.asarray(data_dict['ptr'], dtype=jnp.int32)
        node_mask = self._node_mask_from_data(data_dict)

        raw_counts = ptr[1:] - ptr[:-1]
        batch_size = raw_counts.shape[0]
        max_atoms = species.shape[0]

        graph_index = jnp.arange(batch_size, dtype=jnp.int32)
        batch = jnp.repeat(
            graph_index, raw_counts, total_repeat_length=species.shape[0]
        )
        local_index = jnp.arange(species.shape[0], dtype=jnp.int32) - ptr[batch]
        counts = jax.ops.segment_sum(
            node_mask.astype(jnp.int32),
            batch,
            num_segments=batch_size,
        )
        species = self._remap_species(species)
        species = jnp.where(node_mask, species, -1)

        species_batch = jnp.full((batch_size, max_atoms), -1, dtype=jnp.int32)
        coordinates_batch = jnp.zeros(
            (batch_size, max_atoms, 3),
            dtype=coordinates.dtype,
        )
        atom_mask = jnp.zeros((batch_size, max_atoms), dtype=bool)

        species_batch = species_batch.at[batch, local_index].set(species)
        coordinates_batch = coordinates_batch.at[batch, local_index].set(coordinates)
        atom_mask = atom_mask.at[batch, local_index].set(node_mask)

        force_layout = {
            'batch': batch,
            'local_index': local_index,
            'node_mask': node_mask,
        }

        return species_batch, coordinates_batch, atom_mask, counts, force_layout

    def _flatten_forces(
        self,
        forces: jnp.ndarray,
        atom_mask: jnp.ndarray,
        force_layout: dict[str, jnp.ndarray] | None,
    ) -> jnp.ndarray:
        if force_layout is None:
            return forces[atom_mask]

        gathered = forces[force_layout['batch'], force_layout['local_index']]
        node_mask = jnp.asarray(force_layout['node_mask'], dtype=bool)
        return jnp.where(node_mask[:, None], gathered, 0.0)

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

        apply = getattr(self.module, 'apply')

        if hasattr(self.module, 'init'):
            try:
                return apply(variables, model_inputs)
            except TypeError:
                return apply(variables, species_batch, coordinates_batch)

        try:
            return apply(variables, model_inputs)
        except TypeError:
            pass
        try:
            return apply(variables, species_batch, coordinates_batch)
        except TypeError:
            pass

        apply_fn = apply(variables)
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
        force_layout: dict[str, jnp.ndarray] | None,
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
                result['forces'] = self._flatten_forces(
                    forces,
                    atom_mask,
                    force_layout,
                )
            else:
                result['forces'] = forces

        if 'stress' in result and result['stress'] is not None:
            stress = jnp.asarray(result['stress'])
            if stress.ndim >= 1 and stress.shape[-1] == 6:
                stress = _voigt6_to_full3x3(stress)
            elif stress.ndim == 2 and stress.shape == (3, 3):
                stress = stress[None, :, :]
            result['stress'] = stress[: counts.shape[0]]

        return result

    def apply(
        self,
        variables: dict[str, Any],
        data_dict_or_species: dict[str, jnp.ndarray] | jnp.ndarray,
        coordinates: jnp.ndarray | None = None,
        *,
        compute_force: bool | None = None,
        compute_stress: bool | None = None,
    ) -> dict[str, jnp.ndarray]:
        force_flag = self.compute_force if compute_force is None else compute_force
        stress_flag = self.compute_stress if compute_stress is None else compute_stress

        if coordinates is None:
            if not isinstance(data_dict_or_species, dict):
                raise ValueError(
                    'ANI JAX wrapper expects a data dictionary, or direct '
                    '`species, coordinates` inputs.'
                )
            species_batch, coordinates_batch, atom_mask, counts, force_layout = (
                self._prepare_batch_inputs(data_dict_or_species)
            )
        else:
            species_batch, coordinates_batch, atom_mask, counts, force_layout = (
                self._prepare_direct_inputs(data_dict_or_species, coordinates)
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
                force_layout=force_layout,
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
            result['forces'] = self._flatten_forces(
                forces_full,
                atom_mask,
                force_layout,
            )

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

        species_order = self._species_order_numbers()
        if species_order is None:
            return AtomicNumberTable([1, 6, 7, 8])

        resolved = sorted(set(species_order))
        return AtomicNumberTable(resolved)

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
        # Keep Torch ANI parity: changing the cutoff would require rebuilding the
        # underlying ANI features, so treat this as an intentional no-op.
        return

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
