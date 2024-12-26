import h5py

from ase import Atoms
from torch.utils.data import Dataset
from pathlib import Path

from equitrain.data.configuration import CachedCalc
from equitrain.data.graphs import AtomsToGraphs


class HDF5Dataset(Dataset):

    MAGIC_STRING = "ZVNjaWVuY2UgRXF1aXRyYWlu"

    def __init__(self, filename : Path | str, mode = "r"):
        super().__init__()

        filename = Path(filename)

        if filename.exists():
            self.file = h5py.File(filename, mode)
            self.check_magic()

        else:
            self.file = h5py.File(filename, mode)
            self.write_magic()

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


    def __getitem__(self, i : int):

        grp = self.file[f"i_{i}"]

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
        grp = self.file.create_group(f"i_{i}")
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

    def check_magic(self):
        try:
            grp = self.file["MAGIC"]
            if unpack_value(grp["MAGIC_STRING"][()]) != self.MAGIC_STRING:
                raise IOError("File is not an equitrain data file")

        except KeyError:
            raise IOError("File is not an equitrain data file")


    def write_magic(self):
        grp = self.file.create_group("MAGIC")
        grp["MAGIC_STRING"] = write_value(self.MAGIC_STRING)


def write_value(value):
    return value if value is not None else "None"


def unpack_value(value):
    value = value.decode("utf-8") if isinstance(value, bytes) else value
    return None if str(value) == "None" else value


class HDF5GraphDataset(HDF5Dataset):

    def __init__(self, filename  : Path | str, r_max : float, z_table, mode = "r", **kwargs):
        super().__init__(filename, mode = "r", **kwargs)

        # TODO: Allow users to control what data is returned (i.e. forces, stress)
        self.converter = AtomsToGraphs(z_table, r_energy=True, r_forces=True, r_stress=True, r_pbc=True, radius=r_max)


    def __getitem__(self, index):

        atoms = super(HDF5GraphDataset, self).__getitem__(index)

        return self.converter.convert(atoms)
