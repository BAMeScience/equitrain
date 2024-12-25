
import ase
import numpy as np

from typing import Optional
from dataclasses import dataclass

Vector    = np.ndarray  # [3,]
Positions = np.ndarray  # [..., 3]
Forces    = np.ndarray  # [..., 3]
Stress    = np.ndarray  # [6, ]
Virials   = np.ndarray  # [3,3]
Charges   = np.ndarray  # [..., 1]
Cell      = np.ndarray  # [3,3]
Pbc       = tuple       # (3,)

# Simple data class to convert between Atoms and other data
# types
@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions     : Positions                 # Angstrom
    energy        : Optional[float]   = None  # eV
    forces        : Optional[Forces]  = None  # eV/Angstrom
    stress        : Optional[Stress]  = None  # eV/Angstrom^3
    virials       : Optional[Virials] = None  # eV
    dipole        : Optional[Vector]  = None  # Debye
    charges       : Optional[Charges] = None  # atomic unit
    cell          : Optional[Cell]    = None
    pbc           : Optional[Pbc]     = None

    energy_weight : float = 1.0  # weight of config energy in loss
    forces_weight : float = 1.0  # weight of config forces in loss
    stress_weight : float = 1.0  # weight of config stress in loss
    virials_weight: float = 1.0  # weight of config virial in loss
    dipole_weight : float = 1.0

    @classmethod
    def from_atoms(cls,
        atoms       : ase.Atoms,
        # Keys used in .xyz files
        energy_key  : str  = "energy",
        forces_key  : str  = "forces",
        stress_key  : str  = "stress",
        virials_key : str  = "virials",
        dipole_key  : str  = "dipole",
        charges_key : str  = "charges",
    ) -> "Configuration":

        """Convert ase.Atoms to Configuration"""

        if energy_key == "energy":
            energy = atoms.get_potential_energy()        # eV
        else:
            energy = atoms.info.get(energy_key, None)    # eV

        if forces_key == "forces":
            forces  = atoms.get_forces()                 # eV / Ang
        else:
            forces  = atoms.arrays.get(forces_key, None) # eV / Ang

        if stress_key == "stress":
            stress  = atoms.get_stress()                 # eV / Ang
        else:
            stress  = atoms.info.get(stress_key, None)   # eV / Ang

        virials = atoms.info.get(virials_key, None)
        dipole  = atoms.info.get(dipole_key , None)     # Debye

        # Charges default to 0 instead of None if not found
        charges = atoms.arrays.get(charges_key, np.zeros(len(atoms)))
        atomic_numbers = np.array(
            [ase.data.atomic_numbers[symbol] for symbol in atoms.symbols]
        )
        pbc  = tuple(atoms.get_pbc())
        cell = np.array(atoms.get_cell())

        energy_weight  = 1.0
        forces_weight  = 1.0
        stress_weight  = 1.0
        virials_weight = 1.0
        dipole_weight  = 1.0

        # fill in missing quantities but set weights to zero
        if energy is None:
            energy = 0.0
            energy_weight = 0.0
        if forces is None:
            forces = np.zeros(np.shape(atoms.positions))
            forces_weight = 0.0
        if stress is None:
            stress = np.zeros(6)
            stress_weight = 0.0
        if virials is None:
            virials = np.zeros((3, 3))
            virials_weight = 0.0
        if dipole is None:
            dipole = np.zeros(3)
            dipole_weight = 0.0

        return Configuration(
            atomic_numbers = atomic_numbers,
            positions      = atoms.get_positions(),
            energy         = energy,
            forces         = forces,
            stress         = stress,
            virials        = virials,
            dipole         = dipole,
            charges        = charges,
            pbc            = pbc,
            cell           = cell,
            energy_weight  = energy_weight,
            forces_weight  = forces_weight,
            stress_weight  = stress_weight,
            virials_weight = virials_weight,
            dipole_weight  = dipole_weight,
        )

    def to_atoms(self):

        atoms = ase.Atoms(
            numbers   = self.atomic_numbers,
            positions = self.positions,
            cell      = self.cell,
            pbc       = self.pbc,
        )
        atoms.calc = CachedCalc(
            self.energy,
            self.forces,
            self.stress,
        )
        # Use fixed keys here
        atoms.info  ["virials"] = self.virials
        atoms.info  ["dipole" ] = self.dipole
        atoms.arrays["charges"] = self.charges

        atoms.info  ["energy_weight" ] = self.energy_weight
        atoms.info  ["forces_weight" ] = self.forces_weight
        atoms.info  ["stress_weight" ] = self.stress_weight
        atoms.info  ["virials_weight"] = self.virials_weight
        atoms.info  ["dipole_weight" ] = self.dipole_weight

        return atoms


# Replacement class for ase SinglePointCalculator, which is not
# stable across releases
class CachedCalc:

    def __init__(self, energy, forces, stress):
        self.energy = energy
        self.forces = forces
        self.stress = stress

    def get_potential_energy(self, apply_constraint=False):
        return self.energy

    def get_forces(self, apply_constraint=False):
        return self.forces

    def get_stress(self, apply_constraint=False):
        return self.stress
