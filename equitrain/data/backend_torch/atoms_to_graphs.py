from __future__ import annotations

import numpy as np
import torch
from ase.constraints import FixAtoms
from torch_geometric.data import Data

from equitrain.data.neighborhood import get_neighborhood

from .utility import atomic_numbers_to_indices, to_one_hot


class AtomsToGraphs:
    def __init__(
        self,
        atomic_numbers,
        radius=6,
        r_energy=False,
        r_forces=False,
        r_stress=False,
        r_distances=False,
        r_edges=False,
        r_fixed=False,
        r_pbc=False,
    ):
        self.atomic_numbers = atomic_numbers
        self.radius = radius
        self.r_energy = r_energy
        self.r_forces = r_forces
        self.r_stress = r_stress
        self.r_distances = r_distances
        self.r_fixed = r_fixed
        self.r_edges = r_edges
        self.r_pbc = r_pbc
        self._encoding_cache: dict[
            tuple[int, ...], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def _atomic_encoding(
        self,
        atoms,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = tuple(int(z) for z in atoms.get_atomic_numbers())
        cached = self._encoding_cache.get(key)
        if cached is not None:
            return cached

        atomic_numbers = torch.tensor(key, dtype=torch.long)
        indices = atomic_numbers_to_indices(atomic_numbers, self.atomic_numbers)
        node_attrs = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(self.atomic_numbers),
        )
        self._encoding_cache[key] = (atomic_numbers, node_attrs)
        return atomic_numbers, node_attrs

    def _get_neighbors(self, atoms):
        return get_neighborhood(
            atoms.get_positions(), self.radius, atoms.pbc, np.array(atoms.get_cell())
        )

    def convert(self, atoms):
        atomic_numbers, node_attrs = self._atomic_encoding(atoms)
        dtype = torch.get_default_dtype()
        positions = torch.tensor(atoms.get_positions(), dtype=dtype)
        cell_array = np.array(atoms.get_cell(), dtype=float)
        cell = torch.tensor(cell_array, dtype=dtype).view(1, 3, 3)
        natoms = positions.shape[0]
        tags = torch.Tensor(atoms.get_tags())

        data = Data(
            cell=cell,
            cell_volume=atoms.cell.volume,
            pos=positions,
            positions=positions,
            node_attrs=node_attrs,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
            tags=tags,
            total_charge=torch.tensor(
                _info_float(atoms, 'total_charge', 'charge', default=0.0),
                dtype=dtype,
            ),
            total_spin=torch.tensor(
                _info_float(atoms, 'total_spin', 'spin', default=1.0),
                dtype=dtype,
            ),
            external_field=torch.tensor(
                _external_field(atoms),
                dtype=dtype,
            ).view(1, 3),
            fermi_level=torch.tensor(0.0, dtype=dtype),
            volume=torch.tensor(np.linalg.det(cell_array), dtype=dtype),
            rcell=torch.tensor(_reciprocal_cell(cell_array), dtype=dtype),
        )

        if self.r_edges:
            edge_index, shifts, unit_shifts, cell = self._get_neighbors(atoms)
            if cell is None:
                cell = 3 * [0.0, 0.0, 0.0]

            data.edge_index = torch.tensor(edge_index, dtype=torch.long)
            data.shifts = torch.tensor(shifts, dtype=dtype)
            data.unit_shifts = torch.tensor(unit_shifts, dtype=dtype)
            data.cell = torch.tensor(cell, dtype=dtype)

        if self.r_energy:
            energy = atoms.get_potential_energy(apply_constraint=False)
            data.y = torch.tensor(energy, dtype=dtype)

        if self.r_forces:
            forces = atoms.get_forces(apply_constraint=False)
            data.force = torch.tensor(forces, dtype=dtype)

        if self.r_stress:
            stress = np.array([atoms.get_stress(voigt=False, apply_constraint=False)])
            data.stress = torch.tensor(stress, dtype=dtype)

        if self.r_fixed:
            fixed_idx = torch.zeros(natoms)
            if hasattr(atoms, 'constraints'):
                for constraint in atoms.constraints:
                    if isinstance(constraint, FixAtoms):
                        fixed_idx[constraint.index] = 1
            data.fixed = fixed_idx

        if self.r_pbc:
            data.pbc = torch.tensor(atoms.pbc)

        return data


def _info_float(atoms, *keys, default: float) -> float:
    for key in keys:
        if key in atoms.info:
            return float(np.asarray(atoms.info[key]))
    return default


def _external_field(atoms) -> np.ndarray:
    return np.asarray(
        atoms.info.get('external_field', np.zeros(3, dtype=float)),
        dtype=float,
    ).reshape(3)


def _reciprocal_cell(cell: np.ndarray) -> np.ndarray:
    volume = float(np.linalg.det(cell))
    if abs(volume) <= 1e-10:
        return np.zeros((3, 3), dtype=float)
    return 2.0 * np.pi * np.linalg.inv(cell.T)


__all__ = ['AtomsToGraphs']
