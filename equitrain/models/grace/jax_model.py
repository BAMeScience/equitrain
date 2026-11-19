"""Native JAX GRACE-1L adaptation, 2026-11-18.

Based on tensorpotential by Anton Bochkarev and Yury Lysogorskiy.
Academic Software Licence; see LICENSE.md. No Torch/TensorFlow runtime is used.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .config import GraceConfig
from .topology import angular_tables, mixer_layout, scalar_indices, uncoupled_meta


def spherical_harmonics(unit_vectors, lmax):
    x, y, z = jnp.moveaxis(unit_vectors, -1, 0)
    one = jnp.ones_like(z)
    p = {(0, 0): one}
    if lmax:
        p[1, 0] = math.sqrt(3) * z
        p[1, 1] = -math.sqrt(1.5) * one
    for degree in range(2, lmax + 1):
        for m in range(degree + 1):
            if m == degree:
                p[degree, m] = -math.sqrt(1 + 0.5 / m) * p[degree - 1, m - 1]
            elif m == degree - 1:
                p[degree, m] = math.sqrt(2 * m + 3) * z * p[m, m]
            else:
                a = math.sqrt((4 * degree**2 - 1) / (degree**2 - m**2))
                b = -math.sqrt(((degree - 1) ** 2 - m**2) / (4 * (degree - 1) ** 2 - 1))
                p[degree, m] = a * (z * p[degree - 1, m] + b * p[degree - 2, m])
    phases = [(one, jnp.zeros_like(z))]
    for _ in range(lmax):
        re, im = phases[-1]
        phases.append((re * x - im * y, re * y + im * x))
    values = []
    for degree in range(lmax + 1):
        for m in range(-degree, degree + 1):
            if m == 0:
                values.append(p[degree, 0])
            else:
                values.append(
                    math.sqrt(2)
                    * (-1) ** abs(m)
                    * phases[abs(m)][int(m < 0)]
                    * p[degree, abs(m)]
                )
    return jnp.stack(values, -1)


def chebyshev_radial(distances, cutoff, n_basis, order):
    r = jnp.minimum(distances / cutoff, 1)
    x = 2 * r - 1
    prev, current = jnp.ones_like(x), x
    values = [current]
    for _ in range(1, n_basis):
        prev, current = current, 2 * x * current - prev
        values.append(current)
    polynomial = jnp.full_like(r, order * (order + 1) / 2)
    for degree in range(order - 2, -1, -1):
        polynomial = polynomial * r + (degree + 1) * (degree + 2) / 2
    return jnp.stack(values, -1) * ((1 - r) ** 3 * polynomial)[:, None]


class JaxGrace1L:
    """Functional model with explicit parameter pytree and static angular layout.

    Parameter names and layouts match native Torch GRACE, which permits direct
    numerical comparisons and local weight conversion. Padding masks enter the
    energy computation before differentiation; padded atoms and bonds contribute
    neither energy nor derivatives.
    """

    def __init__(self, config: GraceConfig | dict):
        self.config = (
            config if isinstance(config, GraceConfig) else GraceConfig(**config)
        )
        c = self.config
        self.tables = angular_tables(c.lmax)
        self.l_tile = np.array(
            [degree for degree in range(c.lmax + 1) for _ in range(2 * degree + 1)],
            dtype=np.int32,
        )
        a_meta = uncoupled_meta(c.lmax)
        self.metas = {'A': a_meta}
        self.mixers = {}
        if c.max_order >= 2:
            self.mixers['A1'] = (mixer_layout(a_meta, a_meta), c.n_rad_max, c.n_rad_max)
            self.metas['AA'] = self.tables['AA']['meta']
        for order, name, mixer in [(3, 'AAA', 'AA1'), (4, 'AAAA', 'AA2')]:
            if c.max_order >= order:
                self.mixers[mixer] = (
                    mixer_layout(self.tables['AA']['meta'], a_meta),
                    c.prod_func_n_max,
                    c.n_rad_max,
                )
                self.metas[name] = self.tables[name]['meta']
        self.scalar_indices = {
            name: np.asarray(scalar_indices(meta), dtype=np.int32)
            for name, meta in self.metas.items()
        }
        self.radial_sizes = [c.n_rad_base, *c.radial_hidden, c.n_rad_max * (c.lmax + 1)]
        self.readout_sizes = [c.n_mlp_dens, *c.readout_hidden, 1]

    def parameter_shapes(self):
        c = self.config
        shapes = {
            'embedding': (len(c.atomic_numbers), c.embedding_size),
            'chemical_projection.weight': (c.embedding_size, c.n_rad_max),
        }
        for name, sizes in [
            ('radial', self.radial_sizes),
            ('readout', self.readout_sizes),
        ]:
            for i, (n_in, n_out) in enumerate(zip(sizes, sizes[1:])):
                shapes[f'{name}.layers.{i}.weight'] = (n_in, n_out)
        for name, (layout, n_left, n_right) in self.mixers.items():
            shapes[f'mixers.{name}.weight_left'] = (
                c.prod_func_n_max,
                n_left,
                layout['left_groups'],
            )
            shapes[f'mixers.{name}.weight_right'] = (
                c.prod_func_n_max,
                n_right,
                layout['right_groups'],
            )
        for name, indices in self.scalar_indices.items():
            n_in = c.n_rad_max if name == 'A' else c.prod_func_n_max
            shapes[f'reducers.{name}.weight'] = (
                len(c.atomic_numbers),
                c.n_mlp_dens + 1,
                n_in,
                len(indices),
            )
        shapes['trainable_shift'] = (len(c.atomic_numbers),)
        return shapes

    def init(self, key, dtype=jnp.float32):
        dtype = jnp.dtype(dtype)
        if dtype not in (jnp.dtype('float32'), jnp.dtype('float64')):
            raise ValueError('Native JAX GRACE supports float32 and float64.')
        if dtype == jnp.dtype('float64') and not jax.config.x64_enabled:
            raise ValueError('Enable jax_enable_x64 before initializing float64 GRACE.')
        shapes = self.parameter_shapes()
        keys = jax.random.split(key, len(shapes))
        params = {
            name: (
                jnp.zeros(shape, dtype)
                if name == 'trainable_shift'
                else jax.random.normal(subkey, shape, dtype)
            )
            for (name, shape), subkey in zip(shapes.items(), keys, strict=True)
        }
        return {'params': params}

    def variables_from_torch(self, model, *, dtype=jnp.float64):
        """Copy a local native Torch model; Torch is not imported by this module."""
        dtype = jnp.dtype(dtype)
        if dtype not in (jnp.dtype('float32'), jnp.dtype('float64')):
            raise ValueError('Native JAX GRACE supports float32 and float64.')
        if dtype == jnp.dtype('float64') and not jax.config.x64_enabled:
            raise ValueError('Enable jax_enable_x64 before converting float64 GRACE.')
        if model.config != self.config:
            raise ValueError('Torch and JAX GRACE configurations must match.')
        params = {
            name: jnp.asarray(value.detach().cpu().numpy(), dtype=dtype)
            for name, value in model.named_parameters()
        }
        if set(params) != set(self.parameter_shapes()):
            raise ValueError('Unexpected Torch GRACE parameter names.')
        if any(
            params[name].shape != shape
            for name, shape in self.parameter_shapes().items()
        ):
            raise ValueError('Unexpected Torch GRACE parameter shapes.')
        if not np.allclose(
            model.atomic_energies.detach().cpu(), self.config.atomic_energies
        ):
            raise ValueError(
                'Torch atomic reference energies differ from its configuration.'
            )
        if not math.isclose(model.cutoff.item(), self.config.r_max, rel_tol=1e-6):
            raise ValueError('Torch cutoff differs from its configuration.')
        return {'params': params}

    def validate_variables(self, variables):
        if set(variables) != {'params'} or set(variables['params']) != set(
            self.parameter_shapes()
        ):
            raise ValueError('Unexpected GRACE parameter tree keys.')
        for name, shape in self.parameter_shapes().items():
            value = variables['params'][name]
            if value.shape != shape:
                raise ValueError(
                    f'GRACE parameter {name!r} has shape {value.shape}, expected {shape}.'
                )
            if value.dtype not in (jnp.dtype('float32'), jnp.dtype('float64')):
                raise ValueError(
                    f'GRACE parameter {name!r} must be float32 or float64.'
                )

    @staticmethod
    def _mlp(params, name, sizes, x):
        for i in range(len(sizes) - 1):
            weight = params[f'{name}.layers.{i}.weight']
            x = (x @ weight) * sizes[i] ** -0.5
            if i < len(sizes) - 2:
                x = jax.nn.silu(x)
        return x

    def _mix(self, params, name, left, right):
        layout, n_left, n_right = self.mixers[name]
        left_weight = params[f'mixers.{name}.weight_left'][
            ..., np.asarray(layout['left_tile'])
        ]
        right_weight = params[f'mixers.{name}.weight_right'][
            ..., np.asarray(layout['right_tile'])
        ]
        left = jnp.einsum('onk,ank->aok', left_weight, left) * n_left**-0.5
        right = (
            jnp.einsum(
                'onk,ank->aok',
                right_weight,
                right[..., np.asarray(layout['collect_from'])],
            )
            * n_right**-0.5
        )
        return left.at[..., np.asarray(layout['collect_to'])].add(right) * jnp.asarray(
            layout['output_norm'], left.dtype
        )

    def _product(self, name, left, right):
        table = self.tables[name]
        values = (
            left[..., np.asarray(table['left'])]
            * right[..., np.asarray(table['right'])]
            * jnp.asarray(table['coefficients'], left.dtype)
        )
        result = jnp.zeros((*values.shape[:-1], len(table['meta'])), dtype=left.dtype)
        return result.at[..., np.asarray(table['output'])].add(values)

    def features(
        self,
        variables,
        bond_vectors,
        species,
        centers,
        neighbors,
        *,
        node_mask=None,
        edge_mask=None,
    ):
        c = self.config
        params = variables['params']
        dtype = params['embedding'].dtype
        vectors = jnp.asarray(bond_vectors, dtype=dtype)
        node_mask = (
            jnp.ones(species.shape, bool)
            if node_mask is None
            else jnp.asarray(node_mask, bool)
        )
        edge_mask = (
            jnp.ones(centers.shape, bool)
            if edge_mask is None
            else jnp.asarray(edge_mask, bool)
        )
        safe_species = jnp.clip(species, 0, len(c.atomic_numbers) - 1)
        # Dummy bonds must be finite before any norm/derivative is evaluated.
        vectors = jnp.where(
            edge_mask[:, None], vectors, jnp.array([1.0, 0.0, 0.0], dtype=dtype)
        )
        squared = jnp.sum(vectors**2, -1)
        distances = jnp.sqrt(squared + 1e-10)
        basis = chebyshev_radial(distances, c.r_max, c.n_rad_base, c.cutoff_order)
        radial = self._mlp(params, 'radial', self.radial_sizes, basis)
        radial = radial.reshape((-1, c.n_rad_max, c.lmax + 1))[..., self.l_tile]
        angular = spherical_harmonics(vectors / distances[:, None], c.lmax)
        chemical = (
            params['embedding'] @ params['chemical_projection.weight']
        ) * c.embedding_size**-0.5
        bond_features = (
            radial * angular[:, None, :] * chemical[safe_species[neighbors], :, None]
        )
        bond_features = jnp.where(edge_mask[:, None, None], bond_features, 0.0)
        # Invalid real bonds fail visibly under jit without host callbacks.
        bond_features = jnp.where(
            ((squared > 0) | ~edge_mask)[:, None, None], bond_features, jnp.nan
        )
        a = jnp.zeros((species.shape[0], c.n_rad_max, (c.lmax + 1) ** 2), dtype=dtype)
        a = a.at[centers].add(bond_features) / c.avg_n_neigh
        values = {'RadialBasis': basis, 'R': radial, 'Y': angular, 'A': a}
        if c.max_order >= 2:
            values['A1'] = self._mix(params, 'A1', a, a)
            values['AA'] = self._product('AA', values['A1'], values['A1'])
        if c.max_order >= 3:
            values['AA1'] = self._mix(params, 'AA1', values['AA'], a)
            values['AAA'] = self._product('AAA', values['AA1'], values['A1'])
        if c.max_order >= 4:
            values['AA2'] = self._mix(params, 'AA2', values['AA'], a)
            values['AAAA'] = self._product('AAAA', values['AA2'], values['AA2'])
        rho = jnp.zeros((species.shape[0], c.n_mlp_dens + 1), dtype=dtype)
        for name, indices in self.scalar_indices.items():
            weight = params[f'reducers.{name}.weight'][safe_species]
            selected = values[name][..., indices]
            rho = (
                rho
                + jnp.einsum('aonw,anw->ao', weight, selected)
                * selected.shape[1] ** -0.5
            )
        values['rho'] = rho
        raw = (
            rho[:, 0]
            + self._mlp(params, 'readout', self.readout_sizes, rho[:, 1:])[:, 0]
        )
        atomic = (
            c.energy_scale * raw
            + c.energy_shift
            + jnp.asarray(c.atomic_energies, dtype=dtype)[safe_species]
            + params['trainable_shift'][safe_species]
        )
        valid_species = (species >= 0) & (species < len(c.atomic_numbers))
        values['atomic_energy'] = jnp.where(
            node_mask, jnp.where(valid_species, atomic, jnp.nan), 0.0
        )
        return values

    def apply(self, variables, bond_vectors, species, centers, neighbors, **kwargs):
        return self.features(
            variables, bond_vectors, species, centers, neighbors, **kwargs
        )['atomic_energy']


def save_bundle(output, config, *, seed=0, dtype='float64', variables=None):
    """Write an ordinary Equitrain JAX bundle, with new or supplied local weights."""
    from flax import serialization

    from equitrain.backends.jax_utils import set_jax_dtype

    if dtype not in ('float32', 'float64'):
        raise ValueError('Native JAX GRACE supports float32 and float64.')
    set_jax_dtype(dtype)
    module = JaxGrace1L(config)
    variables = (
        module.init(jax.random.key(seed), getattr(jnp, dtype))
        if variables is None
        else variables
    )
    module.validate_variables(variables)
    variables = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value, dtype=getattr(jnp, dtype)), variables
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'config.json').exists() or (output / 'params.msgpack').exists():
        raise FileExistsError(
            f'Refusing to overwrite an existing model bundle at {output}.'
        )
    payload = asdict(module.config) | {'model_wrapper': 'grace'}
    (output / 'config.json').write_text(json.dumps(payload, indent=2) + '\n')
    (output / 'params.msgpack').write_bytes(serialization.to_bytes(variables))
    return output


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Initialize a local JAX GRACE-1L bundle; no downloads.'
    )
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float64')
    args = parser.parse_args()
    save_bundle(
        args.output,
        GraceConfig(**json.loads(args.config.read_text())),
        seed=args.seed,
        dtype=args.dtype,
    )


if __name__ == '__main__':
    main()
