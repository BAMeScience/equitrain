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

    def _get_neighbors(self, atoms):
        return get_neighborhood(
            atoms.get_positions(), self.radius, atoms.pbc, np.array(atoms.get_cell())
        )

    def convert(self, atoms):
        atomic_numbers = torch.tensor(atoms.get_atomic_numbers())
        positions = torch.tensor(atoms.get_positions(), dtype=torch.get_default_dtype())
        cell = torch.tensor(
            np.array(atoms.get_cell()), dtype=torch.get_default_dtype()
        ).view(1, 3, 3)
        natoms = positions.shape[0]
        tags = torch.Tensor(atoms.get_tags())

        indices = atomic_numbers_to_indices(atomic_numbers, self.atomic_numbers)
        node_attrs = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(self.atomic_numbers),
        )

        data = Data(
            cell=cell,
            cell_volume=atoms.cell.volume,
            pos=positions,
            positions=positions,
            node_attrs=node_attrs,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
            tags=tags,
        )

        if self.r_edges:
            edge_index, shifts, unit_shifts, cell = self._get_neighbors(atoms)
            if cell is None:
                cell = 3 * [0.0, 0.0, 0.0]

            data.edge_index = torch.tensor(edge_index, dtype=torch.long)
            data.shifts = torch.tensor(shifts, dtype=torch.get_default_dtype())
            data.unit_shifts = torch.tensor(
                unit_shifts, dtype=torch.get_default_dtype()
            )
            data.cell = torch.tensor(cell, dtype=torch.get_default_dtype())

        if self.r_energy:
            energy = atoms.get_potential_energy(apply_constraint=False)
            data.y = torch.tensor(energy, dtype=torch.get_default_dtype())

        if self.r_forces:
            forces = atoms.get_forces(apply_constraint=False)
            data.force = torch.tensor(forces, dtype=torch.get_default_dtype())

        if self.r_stress:
            stress = np.array([atoms.get_stress(voigt=False, apply_constraint=False)])
            data.stress = torch.tensor(stress, dtype=torch.get_default_dtype())

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


__all__ = ['AtomsToGraphs']
