"""Native Torch GRACE-1L adaptation, 2026-11-18.

Reference: tensorpotential GRACE_1LAYER_v2_25 by Anton Bochkarev and
Yury Lysogorskiy. Academic Software Licence; see LICENSE.md.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from importlib.resources import files

import torch
from torch import nn

from .layers import (
    MLP,
    NormalizedLinear,
    RightToLeft,
    ScalarReduce,
    TensorProduct,
    chebyshev_radial,
    spherical_harmonics,
)


@dataclass(frozen=True)
class GraceConfig:
    """Supported subset of the upstream GRACE_1LAYER_v2_25 preset.

    Element order is explicit and retained. Energies are in eV and positions
    in angstroms when used with Equitrain's standard data loaders.
    """

    atomic_numbers: tuple[int, ...]
    r_max: float = 6.0
    lmax: int = 4
    max_order: int = 4
    n_rad_base: int = 10
    n_rad_max: int = 32
    prod_func_n_max: int = 64
    embedding_size: int = 128
    n_mlp_dens: int = 16
    radial_hidden: tuple[int, ...] = (64, 64)
    readout_hidden: tuple[int, ...] = (64,)
    cutoff_order: int = 16
    avg_n_neigh: float = 1.0
    energy_scale: float = 1.0
    energy_shift: float = 0.0
    atomic_energies: tuple[float, ...] | None = None

    def __post_init__(self):
        for name in ('atomic_numbers', 'radial_hidden', 'readout_hidden'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (
            not self.atomic_numbers
            or len(set(self.atomic_numbers)) != len(self.atomic_numbers)
            or any(type(z) is not int or not 1 <= z <= 118 for z in self.atomic_numbers)
        ):
            raise ValueError(
                'atomic_numbers must contain distinct integers in [1, 118].'
            )
        for name, lower, upper in [('lmax', 0, 4), ('max_order', 1, 4)]:
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f'{name} must be an integer in [{lower}, {upper}].')
        for name in (
            'n_rad_base',
            'n_rad_max',
            'prod_func_n_max',
            'embedding_size',
            'n_mlp_dens',
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if any(
            type(n) is not int or n < 1
            for n in self.radial_hidden + self.readout_hidden
        ):
            raise ValueError('Hidden layer widths must be positive integers.')
        if type(self.cutoff_order) is not int or self.cutoff_order < 3:
            raise ValueError('cutoff_order must be an integer >= 3.')
        for name in ('r_max', 'avg_n_neigh', 'energy_scale', 'energy_shift'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite.')
        if self.r_max <= 0 or self.avg_n_neigh <= 0:
            raise ValueError('r_max and avg_n_neigh must be positive.')
        if self.atomic_energies is None:
            object.__setattr__(
                self, 'atomic_energies', (0.0,) * len(self.atomic_numbers)
            )
        else:
            object.__setattr__(self, 'atomic_energies', tuple(self.atomic_energies))
        if len(self.atomic_energies) != len(self.atomic_numbers) or not all(
            math.isfinite(x) for x in self.atomic_energies
        ):
            raise ValueError('atomic_energies must have one finite value per element.')
        if self.energy_shift != 0 and any(self.atomic_energies):
            raise ValueError(
                'Use energy_shift or atomic_energies, as in the reference preset.'
            )


class Grace1L(nn.Module):
    """Trainable GRACE-1L with native Torch forward and higher derivatives.

    No upstream package is imported. Static angular tables preserve reference
    pruning, ordering, and normalization. This is not a pretrained model loader.
    """

    def __init__(self, config: GraceConfig | dict):
        super().__init__()
        self.config = (
            config if isinstance(config, GraceConfig) else GraceConfig(**config)
        )
        c = self.config
        n_types = len(c.atomic_numbers)
        self.register_buffer(
            'atomic_numbers', torch.tensor(c.atomic_numbers, dtype=torch.long)
        )
        self.register_buffer(
            'atomic_energies', torch.tensor(c.atomic_energies, dtype=torch.float64)
        )
        self.register_buffer('cutoff', torch.tensor(c.r_max, dtype=torch.float64))
        self.register_buffer(
            'l_tile',
            torch.tensor(
                [degree for degree in range(c.lmax + 1) for _ in range(2 * degree + 1)],
                dtype=torch.long,
            ),
        )
        self.embedding = nn.Parameter(torch.randn(n_types, c.embedding_size))
        self.chemical_projection = NormalizedLinear(c.embedding_size, c.n_rad_max)
        self.radial = MLP([c.n_rad_base, *c.radial_hidden, c.n_rad_max * (c.lmax + 1)])
        a_meta = [
            [degree, m, '', (-1) ** degree, degree]
            for degree in range(c.lmax + 1)
            for m in range(-degree, degree + 1)
        ]
        tables = json.loads(files(__package__).joinpath('couplings.json').read_text())[
            'tables'
        ][str(c.lmax)]
        self.mixers = nn.ModuleDict()
        self.products = nn.ModuleDict()
        self.reducers = nn.ModuleDict(
            {'A': ScalarReduce(a_meta, n_types, c.n_rad_max, c.n_mlp_dens + 1)}
        )
        width = c.prod_func_n_max
        if c.max_order >= 2:
            self.mixers['A1'] = RightToLeft(
                a_meta, a_meta, c.n_rad_max, c.n_rad_max, width
            )
            self.products['AA'] = TensorProduct(tables['AA'])
            self.reducers['AA'] = ScalarReduce(
                tables['AA']['meta'], n_types, width, c.n_mlp_dens + 1
            )
        for order, name, mixer in [(3, 'AAA', 'AA1'), (4, 'AAAA', 'AA2')]:
            if c.max_order >= order:
                self.mixers[mixer] = RightToLeft(
                    tables['AA']['meta'], a_meta, width, c.n_rad_max, width
                )
                self.products[name] = TensorProduct(tables[name])
                self.reducers[name] = ScalarReduce(
                    tables[name]['meta'], n_types, width, c.n_mlp_dens + 1
                )
        self.readout = MLP([c.n_mlp_dens, *c.readout_hidden, 1])
        self.trainable_shift = nn.Parameter(torch.zeros(n_types))

    @classmethod
    def from_json(cls, path):
        with open(path) as handle:
            return cls(GraceConfig(**json.load(handle)))

    def save_config(self, path):
        """Write architecture settings; weights are saved through Torch/Equitrain."""
        with open(path, 'w') as handle:
            json.dump(asdict(self.config), handle, indent=2)
            handle.write('\n')

    def features(self, bond_vectors, species, centers, neighbors):
        """Return named intermediates for reference comparisons and diagnostics.

        Vectors point from center i to neighbor j, including periodic shifts.
        Species are model-local indices; each directed neighbor bond appears once.
        """
        c = self.config
        squared_distances = bond_vectors.square().sum(-1)
        if torch.any(squared_distances <= 0):
            raise ValueError('GRACE requires nonzero bond lengths (overlapping atoms).')
        # Match BondLength's fixed regularizer, including in geometry derivatives.
        distances = torch.sqrt(squared_distances + 1e-10)
        basis = chebyshev_radial(distances, self.cutoff, c.n_rad_base, c.cutoff_order)
        radial = self.radial(basis).reshape(-1, c.n_rad_max, c.lmax + 1)[
            ..., self.l_tile
        ]
        angular = spherical_harmonics(bond_vectors / distances.unsqueeze(-1), c.lmax)
        chemical = self.chemical_projection(self.embedding)[species[neighbors]]
        bond_features = radial * angular.unsqueeze(1) * chemical.unsqueeze(-1)
        a = bond_features.new_zeros((species.shape[0], c.n_rad_max, (c.lmax + 1) ** 2))
        a = a.index_add(0, centers, bond_features) / c.avg_n_neigh
        values = {'RadialBasis': basis, 'R': radial, 'Y': angular, 'A': a}
        if c.max_order >= 2:
            values['A1'] = self.mixers['A1'](a, a)
            values['AA'] = self.products['AA'](values['A1'], values['A1'])
        if c.max_order >= 3:
            values['AA1'] = self.mixers['AA1'](values['AA'], a)
            values['AAA'] = self.products['AAA'](values['AA1'], values['A1'])
        if c.max_order >= 4:
            values['AA2'] = self.mixers['AA2'](values['AA'], a)
            values['AAAA'] = self.products['AAAA'](values['AA2'], values['AA2'])
        rho = sum(
            reducer(values[name], species) for name, reducer in self.reducers.items()
        )
        values['rho'] = rho
        raw_energy = rho[:, 0] + self.readout(rho[:, 1:]).squeeze(-1)
        values['atomic_energy'] = (
            c.energy_scale * raw_energy
            + c.energy_shift
            + self.atomic_energies.to(raw_energy.dtype)[species]
            + self.trainable_shift[species]
        )
        return values

    def forward(self, bond_vectors, species, centers, neighbors):
        return self.features(bond_vectors, species, centers, neighbors)['atomic_energy']
