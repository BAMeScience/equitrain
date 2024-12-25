import h5py

from ase import Atoms
from torch.utils.data import Dataset

from equitrain.data.configuration import CachedCalc
from equitrain.data.graphs import AtomsToGraphs


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
        atoms.info["energy_weight" ] = unpack_value(grp["energy_weight" ][()])
        atoms.info["forces_weight" ] = unpack_value(grp["forces_weight" ][()])
        atoms.info["stress_weight" ] = unpack_value(grp["stress_weight" ][()])
        atoms.info["virials_weight"] = unpack_value(grp["virials_weight"][()])
        atoms.info["dipole_weight" ] = unpack_value(grp["dipole_weight" ][()])

        return atoms


    def __setitem__(self, i : int, atoms : Atoms) -> None:
        grp = self.file.create_group(f"{i}")
        grp["atomic_numbers"] = write_value(atoms.get_atomic_numbers())
        grp["positions"     ] = write_value(atoms.get_positions())
        grp["energy"        ] = write_value(atoms.get_potential_energy())
        grp["forces"        ] = write_value(atoms.get_forces())
        grp["stress"        ] = write_value(atoms.get_stress())
        grp["cell"          ] = write_value(atoms.get_cell())
        grp["pbc"           ] = write_value(atoms.get_pbc())
        grp["virials"       ] = write_value(atoms.info  ["virials"])
        grp["dipole"        ] = write_value(atoms.info  ["dipole" ])
        grp["charges"       ] = write_value(atoms.arrays["charges"])
        grp["energy_weight" ] = write_value(atoms.info  ["energy_weight"])
        grp["forces_weight" ] = write_value(atoms.info  ["forces_weight"])
        grp["stress_weight" ] = write_value(atoms.info  ["stress_weight"])
        grp["virials_weight"] = write_value(atoms.info  ["virials_weight"])
        grp["dipole_weight" ] = write_value(atoms.info  ["dipole_weight"])


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
