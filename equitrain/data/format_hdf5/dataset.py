import h5py
import numpy as np

from ase import Atoms
from torch.utils.data import Dataset
from pathlib import Path

from equitrain.data.configuration import CachedCalc
from equitrain.data.graphs import AtomsToGraphs


class HDF5Dataset:
    MAGIC_STRING = "ZVNjaWVuY2UgRXF1aXRyYWlu"

    def __init__(self, filename: Path | str, mode="r"):
        filename = Path(filename)

        if filename.exists():
            self.file = h5py.File(filename, mode)
            self.check_magic()
        else:
            self.file = h5py.File(filename, mode)
            self.write_magic()
            self.create_dataset()


    def create_dataset(self):
        atom_dtype = np.dtype([
            ("atomic_numbers" , h5py.special_dtype(vlen=np.int32)),
            ("positions"      , h5py.special_dtype(vlen=np.float64)),
            ("cell"           , np.float64, (3, 3)),
            ("pbc"            , np.bool_,   (3,)),
            ("energy"         , np.float64),
            ("forces"         , h5py.special_dtype(vlen=np.float64)),
            ("stress"         , np.float64, (6,)),
            ("virials"        , np.float64, (3, 3)),
            ("dipole"         , np.float64, (3,)),
            ("energy_weight"  , np.float32),
            ("forces_weight"  , np.float32),
            ("stress_weight"  , np.float32),
            ("virials_weight" , np.float32),
            ("dipole_weight"  , np.float32),
        ])
        # There are some parameters that should be accessible through
        # command-line options, i.e. chunking and compression
        self.file.create_dataset(
            "atoms",
            shape    = (0,),        # Initially empty
            maxshape = (None,),     # Extendable along the first dimension
            dtype    = atom_dtype,
        )


    # Allow `with` notation, just syntactic sugar in this case
    def __enter__(self):
        return self


    # Allow `with` notation, just syntactic sugar in this case
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
        if "atoms" not in self.file:
            raise RuntimeError("Dataset 'atoms' does not exist")
        return self.file["atoms"].shape[0]


    def __getitem__(self, i: int) -> Atoms:
        entry = self.file["atoms"][i]
        num_atoms = len(entry["positions"]) // 3
        atoms = Atoms(
            numbers   = entry["atomic_numbers"],
            positions = entry["positions"].reshape((num_atoms, 3)),
            cell      = entry["cell"],
            pbc       = entry["pbc"]
        )
        atoms.calc = CachedCalc(
            entry["energy"],
            entry["forces"].reshape((num_atoms, 3)),
            entry["stress"]
        )
        atoms.info["virials"] = entry["virials"]
        atoms.info["dipole" ] = entry["dipole" ]
        atoms.info["energy_weight" ] = entry["energy_weight" ]
        atoms.info["forces_weight" ] = entry["forces_weight" ]
        atoms.info["stress_weight" ] = entry["stress_weight" ]
        atoms.info["virials_weight"] = entry["virials_weight"]
        atoms.info["dipole_weight" ] = entry["dipole_weight" ]
        return atoms


    def __setitem__(self, i: int, atoms: Atoms) -> None:
        dataset = self.file["atoms"]
        # Extend dataset if necessary
        if i >= len(dataset):
            dataset.resize(i + 1, axis=0)

        dataset[i] = (
            atoms.get_atomic_numbers()     .astype(np.int32),
            atoms.get_positions().flatten().astype(np.float64),
            atoms.get_cell()               .astype(np.float64),
            atoms.get_pbc()                .astype(np.bool_),
            np.float64(atoms.get_potential_energy()),
            atoms.get_forces().flatten()   .astype(np.float64),
            atoms.get_stress()             .astype(np.float64),
            atoms.info["virials"]          .astype(np.float64),
            atoms.info["dipole" ]          .astype(np.float64),
            np.float32(atoms.info.get("energy_weight" , 1.0)),
            np.float32(atoms.info.get("forces_weight" , 1.0)),
            np.float32(atoms.info.get("stress_weight" , 1.0)),
            np.float32(atoms.info.get("virials_weight", 1.0)),
            np.float32(atoms.info.get("dipole_weight" , 1.0)),
        )


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

    def __init__(self, filename  : Path | str, r_max : float, atomic_numbers, mode = "r", **kwargs):
        super().__init__(filename, mode = "r", **kwargs)

        # TODO: Allow users to control what data is returned (i.e. forces, stress)
        self.converter = AtomsToGraphs(atomic_numbers, r_energy=True, r_forces=True, r_stress=True, r_pbc=True, radius=r_max)


    def __getitem__(self, index):

        atoms = super(HDF5GraphDataset, self).__getitem__(index)

        return self.converter.convert(atoms)
