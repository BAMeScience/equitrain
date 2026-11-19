"""Shared GRACE configuration; Academic Software Licence (see LICENSE.md)."""

from __future__ import annotations

import math
from dataclasses import dataclass


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
