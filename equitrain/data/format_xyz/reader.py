
import ase.io
import logging

from typing import Dict, List, Optional, Sequence, Tuple

from equitrain.data import AtomicNumberTable, Configuration, Configurations


class XYZReader():

    def __init__(self,
        filename                : str,
        energy_key              : str  = "energy",
        forces_key              : str  = "forces",
        stress_key              : str  = "stress",
        virials_key             : str  = "virials",
        dipole_key              : str  = "dipole",
        charges_key             : str  = "charges",
        config_type_weights     : Dict = {},
        extract_z_table         : bool = False,
        extract_atomic_energies : bool = False,
        ):

        if energy_key == "energy":
            logging.warning(
                "Since ASE version 3.23.0b1, using energy_key 'energy' is no longer safe when communicating between MACE and ASE. We recommend using a different key, rewriting 'energy' to 'REF_energy'. You need to use --energy_key='REF_energy' to specify the chosen key name."
            )
        if forces_key == "forces":
            logging.warning(
                "Since ASE version 3.23.0b1, using forces_key 'forces' is no longer safe when communicating between MACE and ASE. We recommend using a different key, rewriting 'forces' to 'REF_forces'. You need to use --forces_key='REF_forces' to specify the chosen key name."
            )
        if stress_key == "stress":
            logging.warning(
                "Since ASE version 3.23.0b1, using stress_key 'stress' is no longer safe when communicating between MACE and ASE. We recommend using a different key, rewriting 'stress' to 'REF_stress'. You need to use --stress_key='REF_stress' to specify the chosen key name."
            )

        self.filename                = filename
        self.config_type_weights     = config_type_weights
        self.energy_key              = energy_key
        self.forces_key              = forces_key
        self.stress_key              = stress_key
        self.virials_key             = virials_key
        self.dipole_key              = dipole_key
        self.charges_key             = charges_key
        self.z_set                   = set()
        self.atomic_energies_dict    = {}
        self.extract_z_table         = extract_z_table
        self.extract_atomic_energies = extract_atomic_energies


    def __iter__(self):

        self.atomic_energies_dict = {}

        for i, atoms in enumerate(ase.io.iread(self.filename, index=":")):

            if self.extract_z_table:
                self.z_set.update(atoms.get_atomic_numbers())

            # Do not forward isolated atoms but use this information
            # to update the atomic energies dictionary
            if self.extract_atomic_energies and (len(atoms) == 1 and atoms.info["config_type"] == "IsolatedAtom"):

                self.update_atomic_energies(atoms, i)

            else:

                config = Configuration.from_atoms(
                    atoms,
                    config_type_weights = self.config_type_weights,
                    energy_key          = self.energy_key,
                    forces_key          = self.forces_key,
                    stress_key          = self.stress_key,
                    virials_key         = self.virials_key,
                    dipole_key          = self.dipole_key,
                    charges_key         = self.charges_key,
                )

                yield config


    def update_atomic_energies(self, atoms, i):

        if self.energy_key in atoms.info.keys():
            self.atomic_energies_dict[atoms.get_atomic_numbers()[0]] = atoms.info[
                self.energy_key
            ]
        else:
            logging.warning(
                f"Configuration '{i}' is marked as 'IsolatedAtom' "
                "but does not contain an energy."
            )

    @property
    def z_table(self):
        return AtomicNumberTable(sorted(list(self.z_set)))
