
import ase.io

from typing import Dict, List, Optional, Sequence, Tuple

from equitrain.data import Configurations, process_atoms_list

def load_from_xyz(
    file_path: str,
    config_type_weights: Dict,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipole",
    charges_key: str = "charges",
    extract_atomic_energies: bool = False,
) -> Tuple[Dict[int, float], Configurations]:

    atoms_list = ase.io.read(file_path, index=":")

    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    atomic_energies_dict = {}
    if extract_atomic_energies:
        atoms_without_iso_atoms = []

        for idx, atoms in enumerate(atoms_list):
            if len(atoms) == 1 and atoms.info["config_type"] == "IsolatedAtom":
                if energy_key in atoms.info.keys():
                    atomic_energies_dict[atoms.get_atomic_numbers()[0]] = atoms.info[
                        energy_key
                    ]
                else:
                    logging.warning(
                        f"Configuration '{idx}' is marked as 'IsolatedAtom' "
                        "but does not contain an energy."
                    )
            else:
                atoms_without_iso_atoms.append(atoms)

        if len(atomic_energies_dict) > 0:
            logging.info("Using isolated atom energies from training file")

        atoms_list = atoms_without_iso_atoms

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


def load_from_xyz_in_chunks(
    file_path: str,
    config_type_weights: Dict,
    chunk_size: int = 1000,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipole",
    charges_key: str = "charges",
    extract_atomic_energies: bool = False,
):
    atomic_energies_dict = {}
    total_configs = []
    start = 0

    while True:
        atoms_list = ase.io.read(file_path, index=f"{start}:{start + chunk_size}")

        if not atoms_list:
            break

        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]

        current_energies_dict, configs = process_atoms_list(
            atoms_list,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            virials_key=virials_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            extract_atomic_energies=extract_atomic_energies,
        )


        atomic_energies_dict.update(current_energies_dict)
        total_configs.extend(configs)
        start += chunk_size

    return atomic_energies_dict, total_configs
