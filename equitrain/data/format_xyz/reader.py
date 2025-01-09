import logging

import ase.io

from equitrain.data import AtomicNumberTable, Configuration


class XYZReader:
    def __init__(
        self,
        filename: str,
        energy_key: str = 'energy',
        forces_key: str = 'forces',
        stress_key: str = 'stress',
        virials_key: str = 'virials',
        dipole_key: str = 'dipole',
        charges_key: str = 'charges',
        extract_atomic_numbers: bool = False,
        extract_atomic_energies: bool = False,
    ):
        self.filename = filename
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.stress_key = stress_key
        self.virials_key = virials_key
        self.dipole_key = dipole_key
        self.charges_key = charges_key
        self.z_set = set()
        self.atomic_energies = {}
        self.extract_atomic_numbers = extract_atomic_numbers
        self.extract_atomic_energies = extract_atomic_energies

    def __iter__(self):
        self.atomic_energies = {}

        for i, atoms in enumerate(ase.io.iread(self.filename, index=':')):
            if self.extract_atomic_numbers:
                self.z_set.update([int(z) for z in atoms.get_atomic_numbers()])

            # Do not forward isolated atoms but use this information
            # to update the atomic energies dictionary
            if self.extract_atomic_energies and (
                len(atoms) == 1 and atoms.info['config_type'] == 'IsolatedAtom'
            ):
                self.update_atomic_energies(atoms, i)

            else:
                atoms = Configuration.from_atoms(
                    atoms,
                    energy_key=self.energy_key,
                    forces_key=self.forces_key,
                    stress_key=self.stress_key,
                    virials_key=self.virials_key,
                    dipole_key=self.dipole_key,
                    charges_key=self.charges_key,
                ).to_atoms()

                yield atoms

    def update_atomic_energies(self, atoms, i):
        if self.energy_key in atoms.info.keys():
            self.atomic_energies[atoms.get_atomic_numbers()[0]] = atoms.info[
                self.energy_key
            ]
        else:
            logging.warning(
                f"Configuration '{i}' is marked as 'IsolatedAtom' "
                'but does not contain an energy.'
            )

    @property
    def atomic_numbers(self):
        return AtomicNumberTable(self.z_set)
