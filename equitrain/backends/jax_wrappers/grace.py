"""Native GRACE graph adapter with differentiable JAX forces and stress."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from equitrain.data.atomic import AtomicNumberTable
from equitrain.models.grace import GraceConfig, JaxGrace1L


def _architecture(config):
    metadata = {'model_wrapper', 'wrapper', 'wrapper_name'}
    return GraceConfig(
        **{key: value for key, value in config.items() if key not in metadata}
    )


class GraceWrapper:
    def __init__(self, *, module, config, compute_force=False, compute_stress=False):
        if not isinstance(module, JaxGrace1L):
            raise TypeError('JAX GraceWrapper requires a native JaxGrace1L module.')
        if _architecture(config) != module.config:
            raise ValueError('GRACE bundle configuration does not match the module.')
        self.module = module
        self.config = dict(config)
        self.compute_force = compute_force
        self.compute_stress = compute_stress

    @staticmethod
    def input_dtype(variables):
        return variables['params']['embedding'].dtype

    def apply(self, variables, data_dict, *, compute_force=None, compute_stress=None):
        force = self.compute_force if compute_force is None else compute_force
        stress = self.compute_stress if compute_stress is None else compute_stress
        for key in (
            'positions',
            'node_attrs_index',
            'edge_index',
            'shifts',
            'batch',
            'ptr',
        ):
            if key not in data_dict:
                raise ValueError(f'GRACE JAX input is missing `{key}`.')
        if stress and 'cell' not in data_dict:
            raise ValueError('GRACE JAX stress requires `cell`.')
        dtype = self.input_dtype(variables)
        positions = jnp.asarray(data_dict['positions'], dtype=dtype)
        species = jnp.asarray(data_dict['node_attrs_index'], dtype=jnp.int32)
        centers, neighbors = jnp.asarray(data_dict['edge_index'], dtype=jnp.int32)
        batch = jnp.asarray(data_dict['batch'], dtype=jnp.int32)
        shifts = jnp.asarray(data_dict['shifts'], dtype=dtype)
        node_mask = data_dict.get('node_mask')
        if node_mask is None:
            attrs = data_dict.get('node_attrs')
            node_mask = (
                jnp.any(attrs != 0, axis=-1) if attrs is not None else species >= 0
            )
        node_mask = jnp.asarray(node_mask, dtype=bool)
        edge_mask = jnp.asarray(
            data_dict.get('edge_mask', jnp.ones(centers.shape, bool)), bool
        )
        edge_mask = edge_mask & node_mask[centers] & node_mask[neighbors]
        n_graphs = data_dict['ptr'].shape[0] - 1
        graph_mask = data_dict.get('graph_mask')
        if graph_mask is None:
            graph_mask = (
                jnp.zeros(n_graphs, dtype=jnp.int32)
                .at[batch]
                .add(node_mask.astype(jnp.int32))
                > 0
            )
        graph_mask = jnp.asarray(graph_mask, dtype=bool)

        def energy_fn(coords, strain=None):
            vectors = coords[neighbors] - coords[centers] + shifts
            if strain is not None:
                symmetric = (strain + jnp.swapaxes(strain, -1, -2)) * 0.5
                vectors = vectors + jnp.einsum(
                    'ei,eij->ej', vectors, symmetric[batch[centers]]
                )
            atomic = self.module.apply(
                variables,
                vectors,
                species,
                centers,
                neighbors,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
            energy = jnp.zeros(n_graphs, dtype=dtype).at[batch].add(atomic)
            return jnp.where(graph_mask, energy, 0.0)

        def total(coords, strain=None):
            energy = energy_fn(coords, strain)
            return energy.sum(), energy

        forces = stresses = None
        strain = jnp.zeros((n_graphs, 3, 3), dtype=dtype)
        if force and stress:
            (_, energy), (pos_grad, strain_grad) = jax.value_and_grad(
                total, argnums=(0, 1), has_aux=True
            )(positions, strain)
        elif force:
            (_, energy), pos_grad = jax.value_and_grad(total, has_aux=True)(positions)
        elif stress:
            (_, energy), strain_grad = jax.value_and_grad(
                lambda s: total(positions, s), has_aux=True
            )(strain)
        else:
            energy = energy_fn(positions)
        if force:
            forces = jnp.where(node_mask[:, None], -pos_grad, 0.0)
        if stress:
            cell = jnp.asarray(data_dict['cell'], dtype=dtype).reshape(n_graphs, 3, 3)
            volume = jnp.abs(jnp.linalg.det(cell))
            valid = volume > 1e-12
            # Raise for real invalid cells in eager use; compiled calls produce NaN
            # for invalid real graphs, while padded zero cells remain exactly zero.
            if not isinstance(volume, jax.core.Tracer):
                if np.any(np.asarray(graph_mask) & ~np.asarray(valid)):
                    raise ValueError(
                        'Stress requires a nonzero cell volume for every real structure.'
                    )
            safe_volume = jnp.where(valid, volume, 1.0)
            stresses = jnp.where(
                graph_mask[:, None, None],
                jnp.where(
                    valid[:, None, None],
                    strain_grad / safe_volume[:, None, None],
                    jnp.nan,
                ),
                0.0,
            )
        return {'energy': energy, 'forces': forces, 'stress': stresses}

    @property
    def atomic_numbers(self):
        return AtomicNumberTable(list(self.module.config.atomic_numbers))

    @property
    def atomic_energies(self):
        c = self.module.config
        return [value + c.energy_shift for value in c.atomic_energies]

    @property
    def r_max(self):
        return self.module.config.r_max

    @r_max.setter
    def r_max(self, value):
        if not math.isclose(float(value), self.r_max, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError('Construct a new GRACE config to change its cutoff.')

    def with_compute_flags(self, *, force=None, stress=None):
        if force is not None:
            self.compute_force = force
        if stress is not None:
            self.compute_stress = stress
        return self


def build_module(config):
    module = JaxGrace1L(_architecture(config))
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    return module, module.init(jax.random.key(0), dtype)


__all__ = ['GraceWrapper', 'build_module']
