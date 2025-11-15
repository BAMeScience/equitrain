from pathlib import Path

import h5py
import numpy as np
from ase import Atoms

from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.configuration import CachedCalc, niggli_reduce_inplace


class HDF5Dataset:
    """
    Lightweight, append-only HDF5 store for ASE ``Atoms`` objects.

    Layout & performance
    --------------------
    - ``/structures``: per-configuration metadata (cell, PBC, energy, stress, weights,
      etc.) plus the offset/length that point into the contiguous arrays below.
      Structure-level quantities such as stress or dipole live here because they do
      not scale with atom count.
    - ``/positions``, ``/forces``, ``/atomic_numbers``: flat, chunked arrays that
      contain per-atom data. Random access only touches the slices required for a
      given structure.

    Each per-atom array is chunked by a configurable number of atoms (default 1024),
    which keeps small random reads in-cache for typical batch sizes and avoids the
    pointer-chasing penalty of HDF5 variable-length fields. Appending new structures
    only requires extending those arrays, so the file remains compact and performant
    even with tens of millions of entries.
    """

    MAGIC_STRING = 'ZVNjaWVuY2UgRXF1aXRyYWlu'
    STRUCTURES_DATASET = 'structures'
    POSITIONS_DATASET = 'positions'
    FORCES_DATASET = 'forces'
    ATOMIC_NUMBERS_DATASET = 'atomic_numbers'
    _DEFAULT_CHUNK_ATOMS = 1024

    def __init__(self, filename: Path | str, mode: str = 'r'):
        filename = Path(filename)

        self.file = h5py.File(filename, mode)

        if 'MAGIC' not in self.file:
            self.write_magic()
            self.create_dataset()
        else:
            self.check_magic()
            if self.STRUCTURES_DATASET not in self.file:
                raise OSError(
                    'HDF5 file was created with an unsupported legacy format. '
                    'Please regenerate the dataset with the current Equitrain version.'
                )

    def create_dataset(self):
        if self.STRUCTURES_DATASET in self.file:
            return

        atom_dtype = np.dtype(
            [
                ('offset', np.int64),
                ('length', np.int32),
                ('cell', np.float64, (3, 3)),
                ('pbc', np.bool_, (3,)),
                ('energy', np.float64),
                ('stress', np.float64, (6,)),
                ('virials', np.float64, (3, 3)),
                ('dipole', np.float64, (3,)),
                ('energy_weight', np.float32),
                ('forces_weight', np.float32),
                ('stress_weight', np.float32),
                ('virials_weight', np.float32),
                ('dipole_weight', np.float32),
            ]
        )

        self.file.create_dataset(
            self.STRUCTURES_DATASET,
            shape=(0,),
            maxshape=(None,),
            dtype=atom_dtype,
            chunks=True,
        )

        chunk_atoms = self._DEFAULT_CHUNK_ATOMS
        self.file.create_dataset(
            self.POSITIONS_DATASET,
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=np.float64,
            chunks=(chunk_atoms, 3),
        )
        self.file.create_dataset(
            self.FORCES_DATASET,
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=np.float64,
            chunks=(chunk_atoms, 3),
        )
        self.file.create_dataset(
            self.ATOMIC_NUMBERS_DATASET,
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(chunk_atoms,),
        )

    def open(self, filename: Path | str, mode: str = 'r'):
        """Manually open the dataset file."""
        self.__init__(filename, mode)

    def close(self):
        """Manually close the dataset file."""
        self.file.close()

    def __enter__(self):
        """Enter the runtime context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context."""
        self.close()

    def __del__(self):
        """Ensure the file is closed when the object is deleted."""
        try:
            self.close()
        except Exception:
            # Object was already deleted or invalid
            pass

    def __getstate__(self):
        d = dict(self.__dict__)
        # An opened h5py.File cannot be pickled, so we must exclude it from the state
        d['file'] = None
        return d

    def __len__(self):
        return self.file[self.STRUCTURES_DATASET].shape[0]

    def __getitem__(self, i: int) -> Atoms:
        structures = self.file[self.STRUCTURES_DATASET]
        entry = structures[i]
        offset = int(entry['offset'])
        length = int(entry['length'])
        end = offset + length

        positions = self.file[self.POSITIONS_DATASET][offset:end]
        forces = self.file[self.FORCES_DATASET][offset:end]
        atomic_numbers = self.file[self.ATOMIC_NUMBERS_DATASET][offset:end]

        atoms = Atoms(
            numbers=atomic_numbers.astype(np.int32, copy=False),
            positions=positions,
            cell=entry['cell'],
            pbc=entry['pbc'],
        )
        atoms.calc = CachedCalc(
            float(entry['energy']),
            forces,
            entry['stress'],
        )
        atoms.info['virials'] = entry['virials']
        atoms.info['dipole'] = entry['dipole']
        atoms.info['energy_weight'] = entry['energy_weight']
        atoms.info['forces_weight'] = entry['forces_weight']
        atoms.info['stress_weight'] = entry['stress_weight']
        atoms.info['virials_weight'] = entry['virials_weight']
        atoms.info['dipole_weight'] = entry['dipole_weight']
        return atoms

    def __setitem__(self, i: int, atoms: Atoms) -> None:
        structures = self.file[self.STRUCTURES_DATASET]
        positions_ds = self.file[self.POSITIONS_DATASET]
        forces_ds = self.file[self.FORCES_DATASET]
        atomic_numbers_ds = self.file[self.ATOMIC_NUMBERS_DATASET]

        numbers = atoms.get_atomic_numbers().astype(np.int32, copy=True)
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        length = positions.shape[0]

        if length != numbers.shape[0] or length != forces.shape[0]:
            raise ValueError('Inconsistent atom count for positions/forces/numbers')

        cell = atoms.get_cell().astype(np.float64)
        pbc = atoms.get_pbc().astype(np.bool_)
        energy = np.float64(atoms.get_potential_energy())
        stress = np.asarray(atoms.get_stress(), dtype=np.float64).reshape(6)
        virials = np.asarray(
            atoms.info.get('virials', np.zeros((3, 3), dtype=np.float64)),
            dtype=np.float64,
        ).reshape(3, 3)
        dipole = np.asarray(
            atoms.info.get('dipole', np.zeros(3, dtype=np.float64)),
            dtype=np.float64,
        ).reshape(3)
        energy_weight = np.float32(atoms.info.get('energy_weight', 1.0))
        forces_weight = np.float32(atoms.info.get('forces_weight', 1.0))
        stress_weight = np.float32(atoms.info.get('stress_weight', 1.0))
        virials_weight = np.float32(atoms.info.get('virials_weight', 1.0))
        dipole_weight = np.float32(atoms.info.get('dipole_weight', 1.0))

        current_len = len(structures)
        if i < current_len:
            entry = structures[i]
            if int(entry['length']) != length:
                raise ValueError(
                    'Cannot change number of atoms for an existing entry in HDF5Dataset'
                )
            offset = int(entry['offset'])
            end = offset + length
            positions_ds[offset:end] = positions
            forces_ds[offset:end] = forces
            atomic_numbers_ds[offset:end] = numbers
            structures[i] = (
                offset,
                length,
                cell,
                pbc,
                energy,
                stress,
                virials,
                dipole,
                energy_weight,
                forces_weight,
                stress_weight,
                virials_weight,
                dipole_weight,
            )
            return

        if i > current_len:
            raise IndexError(
                'Cannot assign to non-contiguous index in HDF5Dataset; '
                f'expected index {current_len}, received {i}'
            )

        offset = positions_ds.shape[0]
        end = offset + length

        positions_ds.resize(end, axis=0)
        positions_ds[offset:end] = positions

        forces_ds.resize(end, axis=0)
        forces_ds[offset:end] = forces

        atomic_numbers_ds.resize(end, axis=0)
        atomic_numbers_ds[offset:end] = numbers

        structures.resize(current_len + 1, axis=0)
        structures[current_len] = (
            offset,
            length,
            cell,
            pbc,
            energy,
            stress,
            virials,
            dipole,
            energy_weight,
            forces_weight,
            stress_weight,
            virials_weight,
            dipole_weight,
        )

    def check_magic(self):
        try:
            grp = self.file['MAGIC']
            if unpack_value(grp['MAGIC_STRING'][()]) != self.MAGIC_STRING:
                raise OSError('File is not an equitrain data file')

        except KeyError:
            raise OSError('File is not an equitrain data file')

    def write_magic(self):
        grp = self.file.create_group('MAGIC')
        grp['MAGIC_STRING'] = write_value(self.MAGIC_STRING)


def write_value(value):
    return value if value is not None else 'None'


def unpack_value(value):
    value = value.decode('utf-8') if isinstance(value, bytes) else value
    return None if str(value) == 'None' else value


class HDF5GraphDataset(HDF5Dataset):
    def __init__(
        self,
        filename: Path | str,
        r_max: float,
        atomic_numbers: AtomicNumberTable,
        *,
        niggli_reduce: bool = False,
        atoms_to_graphs_cls=None,
        **kwargs,
    ):
        super().__init__(filename, mode='r', **kwargs)

        self._niggli_reduce = niggli_reduce
        if atoms_to_graphs_cls is None:
            from equitrain.data.backend_torch import (
                AtomsToGraphs as atoms_to_graphs_cls,
            )

        self.converter = atoms_to_graphs_cls(
            atomic_numbers,
            r_edges=True,
            r_energy=True,
            r_forces=True,
            r_stress=True,
            r_pbc=True,
            radius=r_max,
        )

    def __getitem__(self, index):
        atoms = super().__getitem__(index)
        if self._niggli_reduce:
            niggli_reduce_inplace(atoms)
        graph = self.converter.convert(atoms)
        graph.idx = index

        return graph
