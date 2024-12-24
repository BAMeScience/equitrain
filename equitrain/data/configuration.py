
import ase
import numpy as np

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass

Vector    = np.ndarray  # [3,]
Positions = np.ndarray  # [..., 3]
Forces    = np.ndarray  # [..., 3]
Stress    = np.ndarray  # [6, ]
Virials   = np.ndarray  # [3,3]
Charges   = np.ndarray  # [..., 1]
Cell      = np.ndarray  # [3,3]
Pbc       = tuple       # (3,)

DEFAULT_CONFIG_TYPE = "Default"
DEFAULT_CONFIG_TYPE_WEIGHTS = {DEFAULT_CONFIG_TYPE: 1.0}


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

    weight        : float = 1.0  # weight of config in loss
    energy_weight : float = 1.0  # weight of config energy in loss
    forces_weight : float = 1.0  # weight of config forces in loss
    stress_weight : float = 1.0  # weight of config stress in loss
    virials_weight: float = 1.0  # weight of config virial in loss
    config_type   : Optional[str] = DEFAULT_CONFIG_TYPE  # config_type of config


    @classmethod
    def from_atoms(cls,
        atoms              : ase.Atoms,
        energy_key         : str              = "energy",
        forces_key         : str              = "forces",
        stress_key         : str              = "stress",
        virials_key        : str              = "virials",
        dipole_key         : str              = "dipole",
        charges_key        : str              = "charges",
        config_type_weights: Dict[str, float] = None,
    ) -> "Configuration":
        """Convert ase.Atoms to Configuration"""
        if config_type_weights is None:
            config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

        energy  = atoms.info.get(energy_key, None)    # eV
        forces  = atoms.arrays.get(forces_key, None)  # eV / Ang
        stress  = atoms.info.get(stress_key, None)    # eV / Ang
        virials = atoms.info.get(virials_key, None)
        dipole  = atoms.info.get(dipole_key, None)    # Debye

        # Charges default to 0 instead of None if not found
        charges = atoms.arrays.get(charges_key, np.zeros(len(atoms)))
        atomic_numbers = np.array(
            [ase.data.atomic_numbers[symbol] for symbol in atoms.symbols]
        )
        pbc         = tuple(atoms.get_pbc())
        cell        = np.array(atoms.get_cell())
        config_type = atoms.info.get("config_type", "Default")
        weight      = atoms.info.get("config_weight", 1.0) * config_type_weights.get(
            config_type, 1.0
        )
        energy_weight  = atoms.info.get("config_energy_weight" , 1.0)
        forces_weight  = atoms.info.get("config_forces_weight" , 1.0)
        stress_weight  = atoms.info.get("config_stress_weight" , 1.0)
        virials_weight = atoms.info.get("config_virials_weight", 1.0)

        # fill in missing quantities but set their weight to 0.0
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
            # dipoles_weight = 0.0

        return Configuration(
            atomic_numbers = atomic_numbers,
            positions      = atoms.get_positions(),
            energy         = energy,
            forces         = forces,
            stress         = stress,
            virials        = virials,
            dipole         = dipole,
            charges        = charges,
            weight         = weight,
            energy_weight  = energy_weight,
            forces_weight  = forces_weight,
            stress_weight  = stress_weight,
            virials_weight = virials_weight,
            config_type    = config_type,
            pbc            = pbc,
            cell           = cell,
        )


class Configurations:

    def __init__(self, configurations: List[Configuration] = None):
        self._configurations = configurations or []

    def __getitem__(self, index: Union[int, slice]):
        return self._configurations[index]

    def __iter__(self):
        return iter(self._configurations)

    def __len__(self):
        return len(self._configurations)

    def append(self, config: Configuration):
        self._configurations.append(config)

    def extend(self, configs: Iterable[Configuration]):
        """Extend the configurations with an iterable of Configuration objects."""
        self._configurations.extend(configs)

    @classmethod
    def from_atoms_list(cls,
        atoms_list         : List[ase.Atoms],
        energy_key         : str              = "energy",
        forces_key         : str              = "forces",
        stress_key         : str              = "stress",
        virials_key        : str              = "virials",
        dipole_key         : str              = "dipole",
        charges_key        : str              = "charges",
        config_type_weights: Dict[str, float] = None,
    ) -> "Configurations":
        """Convert list of ase.Atoms into Configurations"""
        if config_type_weights is None:
            config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

        all_configs = []

        for atoms in atoms_list:
            all_configs.append(
                Configuration.from_atoms(
                    atoms,
                    energy_key          = energy_key,
                    forces_key          = forces_key,
                    stress_key          = stress_key,
                    virials_key         = virials_key,
                    dipole_key          = dipole_key,
                    charges_key         = charges_key,
                    config_type_weights = config_type_weights,
                )
            )

        return all_configs


@dataclass
class SubsetCollection:
    train: Configurations
    valid: Configurations
    tests: List[Tuple[str, Configurations]]
