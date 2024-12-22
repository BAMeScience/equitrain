
import ase
import numpy as np

from typing import Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass

Vector = np.ndarray  # [3,]
Positions = np.ndarray  # [..., 3]
Forces = np.ndarray  # [..., 3]
Stress = np.ndarray  # [6, ]
Virials = np.ndarray  # [3,3]
Charges = np.ndarray  # [..., 1]
Cell = np.ndarray  # [3,3]
Pbc = tuple  # (3,)

DEFAULT_CONFIG_TYPE = "Default"
DEFAULT_CONFIG_TYPE_WEIGHTS = {DEFAULT_CONFIG_TYPE: 1.0}


@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    energy: Optional[float] = None  # eV
    forces: Optional[Forces] = None  # eV/Angstrom
    stress: Optional[Stress] = None  # eV/Angstrom^3
    virials: Optional[Virials] = None  # eV
    dipole: Optional[Vector] = None  # Debye
    charges: Optional[Charges] = None  # atomic unit
    cell: Optional[Cell] = None
    pbc: Optional[Pbc] = None

    weight: float = 1.0  # weight of config in loss
    energy_weight: float = 1.0  # weight of config energy in loss
    forces_weight: float = 1.0  # weight of config forces in loss
    stress_weight: float = 1.0  # weight of config stress in loss
    virials_weight: float = 1.0  # weight of config virial in loss
    config_type: Optional[str] = DEFAULT_CONFIG_TYPE  # config_type of config


Configurations = List[Configuration]

@dataclass
class SubsetCollection:
    train: Configurations
    valid: Configurations
    tests: List[Tuple[str, Configurations]]


def config_from_atoms_list(
    atoms_list: List[ase.Atoms],
    energy_key="energy",
    forces_key="forces",
    stress_key="stress",
    virials_key="virials",
    dipole_key="dipole",
    charges_key="charges",
    config_type_weights: Dict[str, float] = None,
) -> Configurations:
    """Convert list of ase.Atoms into Configurations"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    all_configs = []
    for atoms in atoms_list:
        all_configs.append(
            config_from_atoms(
                atoms,
                energy_key=energy_key,
                forces_key=forces_key,
                stress_key=stress_key,
                virials_key=virials_key,
                dipole_key=dipole_key,
                charges_key=charges_key,
                config_type_weights=config_type_weights,
            )
        )
    return all_configs


def config_from_atoms(
    atoms: ase.Atoms,
    energy_key="energy",
    forces_key="forces",
    stress_key="stress",
    virials_key="virials",
    dipole_key="dipole",
    charges_key="charges",
    config_type_weights: Dict[str, float] = None,
) -> Configuration:
    """Convert ase.Atoms to Configuration"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    energy = atoms.info.get(energy_key, None)  # eV
    forces = atoms.arrays.get(forces_key, None)  # eV / Ang
    stress = atoms.info.get(stress_key, None)  # eV / Ang
    virials = atoms.info.get(virials_key, None)
    dipole = atoms.info.get(dipole_key, None)  # Debye
    # Charges default to 0 instead of None if not found
    charges = atoms.arrays.get(charges_key, np.zeros(len(atoms)))  # atomic unit
    atomic_numbers = np.array(
        [ase.data.atomic_numbers[symbol] for symbol in atoms.symbols]
    )
    pbc = tuple(atoms.get_pbc())
    cell = np.array(atoms.get_cell())
    config_type = atoms.info.get("config_type", "Default")
    weight = atoms.info.get("config_weight", 1.0) * config_type_weights.get(
        config_type, 1.0
    )
    energy_weight = atoms.info.get("config_energy_weight", 1.0)
    forces_weight = atoms.info.get("config_forces_weight", 1.0)
    stress_weight = atoms.info.get("config_stress_weight", 1.0)
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
        atomic_numbers=atomic_numbers,
        positions=atoms.get_positions(),
        energy=energy,
        forces=forces,
        stress=stress,
        virials=virials,
        dipole=dipole,
        charges=charges,
        weight=weight,
        energy_weight=energy_weight,
        forces_weight=forces_weight,
        stress_weight=stress_weight,
        virials_weight=virials_weight,
        config_type=config_type,
        pbc=pbc,
        cell=cell,
    )


def process_atoms_list(
    atoms_list,
    config_type_weights,
    energy_key="energy",
    forces_key="forces",
    stress_key="stress",
    virials_key="virials",
    dipole_key="dipole",
    charges_key="charges",
    extract_atomic_energies=False
):
    atomic_energies_dict = {}

    for atoms in atoms_list:
        unique_atomic_numbers = set(atoms.get_atomic_numbers())
        for atomic_number in unique_atomic_numbers:
            if atomic_number not in atomic_energies_dict:
                atomic_energies_dict[atomic_number] = 0.0  # Initialize with None or a default


    # Iterate over all configurations in the atoms_list
    for idx, atoms in enumerate(atoms_list):
        unique_atomic_numbers = set(atoms.get_atomic_numbers())
        energy_key = energy_key if energy_key in atoms.info else "energy_corrected"

        if energy_key in atoms.info:
            energy_value = atoms.info[energy_key]
            atomic_energies_dict[idx] = energy_value
        else:
            logging.error(f"Configuration '{idx}' does not contain an '{energy_key}' field.")
    
    # Process configurations with the extracted atomic energies
    configs = config_from_atoms_list(
        atoms_list,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
    )

    return atomic_energies_dict, configs


def test_config_types(
    test_configs: Configurations,
) -> List[Tuple[Optional[str], List[Configuration]]]:
    """Split test set based on config_type-s"""
    test_by_ct = []
    all_cts = []
    for conf in test_configs:
        if conf.config_type not in all_cts:
            all_cts.append(conf.config_type)
            test_by_ct.append((conf.config_type, [conf]))
        else:
            ind = all_cts.index(conf.config_type)
            test_by_ct[ind][1].append(conf)
    return test_by_ct
