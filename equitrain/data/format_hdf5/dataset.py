import h5py

from ase import Atoms
from torch.utils.data import Dataset, IterableDataset, ChainDataset

from equitrain.data        import Configuration
from equitrain.data.graphs import AtomsToGraphs


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


class HDF5Dataset(Dataset):
    def __init__(self, filename, mode = "r", **kwargs):
        super().__init__()
        self.filename   = filename
        self.mode       = mode
        self.file       = h5py.File(self.filename, self.mode)


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        pass


    def __del__(self):
        """Ensure the file is closed when the object is deleted."""
        if self.file:
            self.file.close()


    def __getstate__(self):
        d = dict(self.__dict__)
        # An opened h5py.File cannot be pickled, so we must exclude it from the state
        d["file"] = None
        return d


    def __len__(self):
        return len(self.file.keys())


    def __getitem__(self, index):

        grp = self.file[str(index)]

        atoms = Atoms(
            numbers   = grp["atomic_numbers"][()],
            positions = grp["positions"][()],
            cell      = unpack_value(grp["cell"][()]),
            pbc       = unpack_value(grp["pbc"][()]),
        )
        atoms.calc = CachedCalc(
            unpack_value(grp["energy"][()]),
            unpack_value(grp["forces"][()]),
            unpack_value(grp["stress"][()]),
        )

        return atoms


    def save_configuration(self, config : Configuration, i : int) -> None:
        grp = self.file.create_group(f"{i}")
        grp["atomic_numbers"] = write_value(config.atomic_numbers)
        grp["positions"     ] = write_value(config.positions)
        grp["energy"        ] = write_value(config.energy)
        grp["forces"        ] = write_value(config.forces)
        grp["stress"        ] = write_value(config.stress)
        grp["virials"       ] = write_value(config.virials)
        grp["dipole"        ] = write_value(config.dipole)
        grp["charges"       ] = write_value(config.charges)
        grp["cell"          ] = write_value(config.cell)
        grp["pbc"           ] = write_value(config.pbc)
        grp["weight"        ] = write_value(config.weight)
        grp["energy_weight" ] = write_value(config.energy_weight)
        grp["forces_weight" ] = write_value(config.forces_weight)
        grp["stress_weight" ] = write_value(config.stress_weight)
        grp["virials_weight"] = write_value(config.virials_weight)
        grp["config_type"   ] = write_value(config.config_type)


def write_value(value):
    return value if value is not None else "None"


def unpack_value(value):
    value = value.decode("utf-8") if isinstance(value, bytes) else value
    return None if str(value) == "None" else value


class HDF5GraphDataset(HDF5Dataset):
    def __init__(self, filename, r_max, z_table, mode = "r", **kwargs):
        super().__init__(filename, mode = "r", **kwargs)

        self.converter = AtomsToGraphs(z_table, r_energy=True, r_forces=True, r_stress=True, r_pbc=True, radius=r_max)


    def __getitem__(self, index):

        atoms = super(HDF5GraphDataset, self).__getitem__(index)

        return self.converter.convert(atoms)
