from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.geometry import cellpar_to_cell


class _AseDbImportError(RuntimeError):
    pass


def _load_aselmdb_dataset():
    try:
        from fairchem.core.datasets import AseDBDataset  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise _AseDbImportError(
            'The optional dependency `fairchem` is required to read ASE-LMDB datasets. '
            'Install it (e.g. `pip install fairchem`) to enable LMDB conversion.'
        ) from exc
    return AseDBDataset


from equitrain.data.format_hdf5.dataset import HDF5Dataset


def _coerce_cell(cell_array: np.ndarray) -> np.ndarray:
    """Normalise cell information to a 3x3 matrix."""
    cell_array = np.asarray(cell_array, dtype=float)
    if cell_array.size == 0:
        return np.zeros((3, 3), dtype=float)
    if cell_array.shape == (3, 3):
        return cell_array
    flat = cell_array.flatten()
    if flat.size == 9:
        return flat.reshape(3, 3)
    if flat.size == 6:
        return cellpar_to_cell(flat)
    if flat.size == 3:
        # Interpret as orthorhombic lengths.
        return np.diagflat(flat)
    raise ValueError(f'Unsupported cell description with shape {cell_array.shape!r}.')


def _coerce_pbc(pbc: Iterable[bool] | None) -> np.ndarray:
    values = np.asarray(
        list(pbc) if pbc is not None else [True, True, True], dtype=bool
    )
    if values.size != 3:
        raise ValueError(f'Expected three PBC flags, received {values}.')
    return values


def _coerce_forces(forces: np.ndarray | None, num_atoms: int) -> np.ndarray:
    if forces is None:
        return np.zeros((num_atoms, 3), dtype=float)
    array = np.asarray(forces, dtype=float)
    return array.reshape(num_atoms, 3)


def _voigt_3x3_to_6(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3):
        raise ValueError('Expected a 3x3 matrix for Voigt conversion.')
    xx, yy, zz = m[0, 0], m[1, 1], m[2, 2]
    yz, xz, xy = m[1, 2], m[0, 2], m[0, 1]
    return np.array([xx, yy, zz, yz, xz, xy], dtype=float)


def _coerce_stress(stress: np.ndarray | None) -> np.ndarray:
    if stress is None:
        return np.zeros(6, dtype=float)
    array = np.asarray(stress, dtype=float).flatten()
    if array.size == 6:
        return array
    if array.size == 9:
        return _voigt_3x3_to_6(array.reshape(3, 3))
    raise ValueError(f'Unsupported stress representation with shape {array.shape!r}.')


def lmdb_entry_to_atoms(entry: Mapping) -> Atoms:
    """Convert a single LMDB record to an ASE ``Atoms`` instance with a single-point calculator."""
    positions = np.asarray(entry['pos'], dtype=float)
    atomic_numbers = np.asarray(entry['atomic_numbers'], dtype=int)
    cell = _coerce_cell(np.asarray(entry.get('cell', np.eye(3, dtype=float))))
    atoms = Atoms(numbers=atomic_numbers, positions=positions, cell=cell)
    atoms.pbc = _coerce_pbc(entry.get('pbc'))

    energy = float(np.asarray(entry.get('energy', 0.0), dtype=float))
    forces = _coerce_forces(entry.get('forces'), len(atoms))
    stress = _coerce_stress(entry.get('stress'))

    calculator = SinglePointCalculator(
        atoms, energy=energy, forces=forces, stress=stress
    )
    atoms.calc = calculator

    # Populate auxiliary metadata expected by the HDF5 writer.
    atoms.info.setdefault('virials', np.zeros((3, 3), dtype=float))
    atoms.info.setdefault('dipole', np.zeros(3, dtype=float))
    atoms.info.setdefault('energy_weight', 1.0)
    atoms.info.setdefault('forces_weight', 1.0)
    atoms.info.setdefault('stress_weight', 1.0)
    atoms.info.setdefault('virials_weight', 1.0)
    atoms.info.setdefault('dipole_weight', 1.0)

    return atoms


def iter_lmdb_atoms(
    src: Path | str,
    *,
    config: Mapping | None = None,
) -> Iterator[Atoms]:
    """Yield ``Atoms`` instances from an ASE-LMDB dataset."""
    AseDBDataset = _load_aselmdb_dataset()
    AseDBDataset = _load_aselmdb_dataset()
    dataset = AseDBDataset(
        config=dict(src=str(src), **(dict(config) if config else {}))
    )
    for record in dataset:
        yield lmdb_entry_to_atoms(record)


def convert_lmdb_to_hdf5(
    src: Path | str,
    dst: Path | str,
    *,
    config: Mapping | None = None,
    overwrite: bool = False,
    show_progress: bool = False,
) -> Path:
    """
    Convert an ASE-LMDB dataset to the EquiTrain HDF5 format.

    Parameters
    ----------
    src:
        Path to the LMDB dataset directory/file recognised by ``AseDBDataset``.
    dst:
        Path to the destination HDF5 file. Existing files are overwritten when
        ``overwrite`` is ``True``.
    config:
        Optional dictionary passed to ``AseDBDataset`` (e.g. metadata entries).
    overwrite:
        When ``False`` (default) an existing destination file raises ``FileExistsError``.
    show_progress:
        Emit a textual progress bar using ``tqdm`` when available.
    """

    src = Path(src)
    dst = Path(dst)

    if dst.exists() and not overwrite:
        raise FileExistsError(f'Destination file {dst} already exists.')

    AseDBDataset = _load_aselmdb_dataset()
    dataset = AseDBDataset(
        config=dict(src=str(src), **(dict(config) if config else {}))
    )
    iterator: Iterable = dataset

    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=len(dataset), desc='Converting LMDB')
        except Exception:  # pragma: no cover - optional dependency
            iterator = dataset

    with HDF5Dataset(dst, mode='w') as storage:
        for index, record in enumerate(iterator):
            atoms = lmdb_entry_to_atoms(record)
            storage[index] = atoms

    return dst


__all__ = ['lmdb_entry_to_atoms', 'iter_lmdb_atoms', 'convert_lmdb_to_hdf5']
