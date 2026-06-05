"""
JAX M3GNet wrapper.
"""

from importlib import import_module
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS

from equitrain.backends.jax_runtime import ensure_multiprocessing_spawn
from equitrain.data.atomic import AtomicNumberTable

ensure_multiprocessing_spawn()


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


def _cell_volume(cell: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum(
        'gi,gi->g', cell[:, 0, :], jnp.cross(cell[:, 1, :], cell[:, 2, :])
    )


def _apply_displacement(
    positions: jnp.ndarray,
    batch: jnp.ndarray,
    displacement: jnp.ndarray,
) -> jnp.ndarray:
    symmetric = 0.5 * (displacement + jnp.swapaxes(displacement, -1, -2))
    return positions + jnp.einsum('ni,nij->nj', positions, symmetric[batch])


def _import_symbol(path: str):
    module_name, _, attr_name = path.replace(':', '.').rpartition('.')
    if not module_name or not attr_name:
        raise ValueError(
            f'Invalid import path "{path}". Expected "package.module:attribute".'
        )
    module = import_module(module_name)
    return getattr(module, attr_name)


class M3GNetWrapper:
    """
    Thin wrapper around a JAX M3GNet-style graph module.

    The wrapped module is expected to accept a mapping with flat graph tensors
    matching Equitrain's JAX graph representation. The wrapper supplies both
    Equitrain names and MatGL-like aliases:

    - ``positions`` / ``pos``
    - ``node_attrs_index`` / ``node_type`` / ``species``
    - ``edge_index`` plus ``senders`` / ``receivers``
    - ``shifts`` / ``pbc_offshift``
    - ``unit_shifts`` / ``pbc_offset``

    The module output must include graph energies and may include forces and
    stress. If forces are requested but absent, the wrapper computes them from
    the energy with ``jax.grad``.
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

    def _model_atomic_numbers(self) -> list[int] | None:
        for key in (
            'model_atomic_numbers',
            'supported_atomic_numbers',
            'm3gnet_atomic_numbers',
        ):
            numbers = self.config.get(key)
            if numbers is not None:
                return [int(z) for z in numbers]

        element_types = self.config.get('element_types')
        if element_types is None:
            return None

        resolved = []
        for symbol in element_types:
            if symbol not in ASE_ATOMIC_NUMBERS:
                raise ValueError(f'Unknown chemical symbol: {symbol}')
            resolved.append(int(ASE_ATOMIC_NUMBERS[symbol]))
        return resolved

    def _species_index_remap(self) -> jnp.ndarray | None:
        dataset_numbers = self.config.get('atomic_numbers')
        model_numbers = self._model_atomic_numbers()

        if dataset_numbers is None or model_numbers is None:
            return None

        dataset_numbers = [int(z) for z in dataset_numbers]
        model_numbers = [int(z) for z in model_numbers]
        if dataset_numbers == model_numbers:
            return None

        index_by_number = {int(z): idx for idx, z in enumerate(model_numbers)}
        remap = []
        for number in dataset_numbers:
            if number not in index_by_number:
                raise ValueError(
                    'M3GNet model element set does not cover every atomic number '
                    f'in config.atomic_numbers: missing Z={number}.'
                )
            remap.append(index_by_number[number])
        return jnp.asarray(remap, dtype=jnp.int32)

    def _remap_species(self, species: jnp.ndarray) -> jnp.ndarray:
        remap = self._species_index_remap()
        if remap is None:
            return species

        valid = species >= 0
        in_range = species < remap.shape[0]
        safe_species = jnp.where(valid & in_range, species, 0)
        remapped = remap[safe_species]
        return jnp.where(valid & in_range, remapped, -1)

    def _node_mask_from_data(self, data_dict: dict[str, jnp.ndarray]) -> jnp.ndarray:
        if 'node_mask' in data_dict:
            return jnp.asarray(data_dict['node_mask'], dtype=bool)
        if 'node_attrs' in data_dict:
            node_attrs = jnp.asarray(data_dict['node_attrs'])
            return jnp.sum(jnp.abs(node_attrs), axis=-1) > 0
        return jnp.ones((jnp.asarray(data_dict['positions']).shape[0],), dtype=bool)

    def _prepare_inputs(
        self,
        data_dict: dict[str, jnp.ndarray],
        positions: jnp.ndarray | None = None,
    ) -> tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
        if not isinstance(data_dict, dict):
            raise ValueError('M3GNet JAX wrapper expects a data dictionary input.')
        for key in ('positions', 'node_attrs_index', 'edge_index', 'ptr'):
            if key not in data_dict:
                raise ValueError(
                    f'M3GNet JAX wrapper input is missing required key `{key}`.'
                )

        coords = (
            jnp.asarray(data_dict['positions'])
            if positions is None
            else jnp.asarray(positions)
        )
        node_type = self._remap_species(
            jnp.asarray(data_dict['node_attrs_index'], dtype=jnp.int32)
        )
        edge_index = jnp.asarray(data_dict['edge_index'], dtype=jnp.int32)
        senders = edge_index[0]
        receivers = edge_index[1]
        ptr = jnp.asarray(data_dict['ptr'], dtype=jnp.int32)

        node_mask = self._node_mask_from_data(data_dict)
        node_type = jnp.where(node_mask, node_type, -1)
        coords = jnp.where(node_mask[:, None], coords, 0.0)

        if 'shifts' in data_dict:
            shifts = jnp.asarray(data_dict['shifts'], dtype=coords.dtype)
        else:
            shifts = jnp.zeros((senders.shape[0], 3), dtype=coords.dtype)

        if 'unit_shifts' in data_dict:
            unit_shifts = jnp.asarray(data_dict['unit_shifts'], dtype=coords.dtype)
        else:
            unit_shifts = jnp.zeros((senders.shape[0], 3), dtype=coords.dtype)

        if 'batch' in data_dict:
            batch = jnp.asarray(data_dict['batch'], dtype=jnp.int32)
        else:
            batch = jnp.repeat(
                jnp.arange(ptr.shape[0] - 1, dtype=jnp.int32),
                ptr[1:] - ptr[:-1],
                total_repeat_length=coords.shape[0],
            )

        if 'cell' in data_dict:
            cell = jnp.asarray(data_dict['cell'], dtype=coords.dtype)
        else:
            cell = jnp.zeros((ptr.shape[0] - 1, 3, 3), dtype=coords.dtype)
        edge_batch = batch[senders]

        inputs = dict(data_dict)
        inputs.update(
            {
                'positions': coords,
                'pos': coords,
                'node_attrs_index': node_type,
                'node_type': node_type,
                'species': node_type,
                'edge_index': edge_index,
                'senders': senders,
                'receivers': receivers,
                'shifts': shifts,
                'unit_shifts': unit_shifts,
                'pbc_offshift': shifts,
                'pbc_offset': unit_shifts,
                'batch': batch,
                'edge_batch': edge_batch,
                'ptr': ptr,
                'cell': cell,
                'node_mask': node_mask,
                'graph_mask': (ptr[1:] - ptr[:-1]) > 0,
            }
        )
        return inputs, coords, node_mask

    def _call_module(
        self,
        variables: dict[str, Any],
        inputs: dict[str, jnp.ndarray],
        *,
        compute_force: bool,
        compute_stress: bool,
    ):
        apply = getattr(self.module, 'apply')

        if hasattr(self.module, 'init'):
            try:
                return apply(
                    variables,
                    inputs,
                    compute_force=compute_force,
                    compute_stress=compute_stress,
                )
            except TypeError:
                return apply(variables, inputs)

        try:
            return apply(
                variables,
                inputs,
                compute_force=compute_force,
                compute_stress=compute_stress,
            )
        except TypeError:
            pass
        try:
            return apply(variables, inputs)
        except TypeError:
            pass

        apply_fn = apply(variables)
        try:
            outputs = apply_fn(
                inputs,
                compute_force=compute_force,
                compute_stress=compute_stress,
            )
        except TypeError:
            outputs = apply_fn(inputs)

        if isinstance(outputs, tuple) and len(outputs) == 2:
            return outputs[0]
        return outputs

    def _flatten_dense_forces(
        self,
        forces: jnp.ndarray,
        data_dict: dict[str, jnp.ndarray],
        node_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        if forces.ndim != 3:
            return forces

        ptr = jnp.asarray(data_dict['ptr'], dtype=jnp.int32)
        batch = jnp.asarray(data_dict['batch'], dtype=jnp.int32)
        local_index = jnp.arange(batch.shape[0], dtype=jnp.int32) - ptr[batch]
        gathered = forces[batch, local_index]
        return jnp.where(node_mask[:, None], gathered, 0.0)

    def _normalize_outputs(
        self,
        outputs,
        *,
        data_dict: dict[str, jnp.ndarray],
        node_mask: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        n_graphs = jnp.asarray(data_dict['ptr']).shape[0] - 1

        if isinstance(outputs, dict):
            result = dict(outputs)
        elif hasattr(outputs, 'energy') or hasattr(outputs, 'energies'):
            if hasattr(outputs, 'energy'):
                energy = getattr(outputs, 'energy')
            else:
                energy = getattr(outputs, 'energies')
            result = {'energy': energy}
            if hasattr(outputs, 'forces'):
                result['forces'] = getattr(outputs, 'forces')
            if hasattr(outputs, 'stress'):
                result['stress'] = getattr(outputs, 'stress')
        elif isinstance(outputs, list | tuple):
            if len(outputs) < 1:
                raise ValueError('M3GNet module returned an empty tuple/list.')
            result = {'energy': outputs[0]}
            if len(outputs) > 1:
                result['forces'] = outputs[1]
            if len(outputs) > 2:
                result['stress'] = outputs[2]
        else:
            result = {'energy': outputs}

        energy = result.get('energy', result.get('energies'))
        if energy is None:
            raise ValueError('M3GNet module output must contain energies.')
        result['energy'] = jnp.asarray(energy).reshape(-1)[:n_graphs]

        if 'forces' in result and result['forces'] is not None:
            forces = jnp.asarray(result['forces'])
            forces = self._flatten_dense_forces(forces, data_dict, node_mask)
            result['forces'] = jnp.where(node_mask[:, None], forces, 0.0)

        if 'stress' in result and result['stress'] is not None:
            stress = jnp.asarray(result['stress'])
            if stress.ndim >= 1 and stress.shape[-1] == 6:
                stress = _voigt6_to_full3x3(stress)
            elif stress.ndim == 2 and stress.shape == (3, 3):
                stress = stress[None, :, :]
            result['stress'] = stress[:n_graphs]

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

        if stress_flag and 'cell' not in data_dict:
            raise ValueError(
                'M3GNet JAX wrapper requires `cell` in the input data when '
                'stress computation is requested.'
            )

        base_inputs, positions, node_mask = self._prepare_inputs(data_dict)

        def _forward(coords, displacement=None):
            if displacement is None:
                model_positions = coords
            else:
                model_positions = _apply_displacement(
                    coords,
                    base_inputs['batch'],
                    displacement,
                )
            inputs, _, active_node_mask = self._prepare_inputs(
                data_dict,
                model_positions,
            )
            outputs = self._call_module(
                variables,
                inputs,
                compute_force=force_flag,
                compute_stress=stress_flag,
            )
            return self._normalize_outputs(
                outputs,
                data_dict=inputs,
                node_mask=active_node_mask,
            )

        normalized = _forward(positions)
        result = {
            'energy': normalized['energy'],
            'forces': normalized.get('forces'),
            'stress': normalized.get('stress'),
        }

        if force_flag and result['forces'] is None:

            def _total_energy(coords):
                return jnp.sum(_forward(coords)['energy'])

            forces = -jax.grad(_total_energy)(positions)
            result['forces'] = jnp.where(node_mask[:, None], forces, 0.0)

        if stress_flag and result['stress'] is None:
            n_graphs = base_inputs['ptr'].shape[0] - 1
            displacement = jnp.zeros((n_graphs, 3, 3), dtype=positions.dtype)

            def _total_energy_for_displacement(disp):
                return jnp.sum(_forward(positions, disp)['energy'])

            virials = jax.grad(_total_energy_for_displacement)(displacement)
            volume = _cell_volume(base_inputs['cell'])
            safe_volume = jnp.where(
                jnp.abs(volume) > jnp.finfo(positions.dtype).eps,
                volume,
                1.0,
            )
            result['stress'] = virials / safe_volume[:, None, None]

        return result

    @property
    def atomic_numbers(self) -> AtomicNumberTable | None:
        numbers = self.config.get('atomic_numbers')
        if numbers is not None:
            return AtomicNumberTable([int(z) for z in numbers])

        model_numbers = self._model_atomic_numbers()
        if model_numbers is None:
            return None
        return AtomicNumberTable(model_numbers)

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
    def r_max(self) -> float | None:
        value = self.config.get('r_max', self.config.get('cutoff'))
        return float(value) if value is not None else None

    @r_max.setter
    def r_max(self, value: float):
        self.config['r_max'] = float(value)
        self.config['cutoff'] = float(value)

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
            'M3GNet JAX config must define `module_factory`, '
            '`module_builder`, or `module_class`.'
        )

    if isinstance(module, tuple):
        if len(module) == 2:
            return module
        raise ValueError(
            'M3GNet JAX module factory must return either `module` or '
            '`(module, params_template)`. '
            f'Got a tuple with {len(module)} entries.'
        )

    return module, None


__all__ = ['M3GNetWrapper', 'build_module']
