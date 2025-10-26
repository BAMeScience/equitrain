import ast
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .atomic import AtomicNumberTable


@dataclass
class Statistics:
    atomic_numbers: AtomicNumberTable = None
    atomic_energies: dict[int, float] = None

    # Energy statistics
    mean: float = None
    std: float = None

    avg_num_neighbors: float = None

    r_max: float = None

    @classmethod
    def load(cls, filename: Path | str) -> dict:
        with open(filename) as f:
            statistics_dict = json.load(f)

        statistics = cls(**statistics_dict)
        # Convert atomic numbers list
        statistics.atomic_numbers = AtomicNumberTable(statistics.atomic_numbers)
        # JSON requires dict keys to be quoted, convert back to integers
        statistics.atomic_energies = {
            int(key): float(value) for key, value in statistics.atomic_energies.items()
        }

        return statistics

    def dump(self, filename: Path | str):
        with open(filename, 'w') as f:
            json.dump(asdict(self), f, indent=4)


def get_atomic_energies(E0s, dataset, z_table) -> dict:
    if E0s is not None:
        logging.log(
            1, 'Atomic Energies not in training file, using command line argument E0s'
        )
        if E0s.lower() == 'average':
            logging.log(
                1, 'Computing average Atomic Energies using least squares regression'
            )
            # catch if colections.train not defined above
            try:
                assert dataset is not None
                atomic_energies_dict = compute_average_atomic_energies(dataset, z_table)
            except Exception as e:
                raise RuntimeError(
                    f'Could not compute average E0s if no training xyz given, error {e} occured'
                ) from e
        else:
            try:
                atomic_energies_dict = ast.literal_eval(E0s)
                assert isinstance(atomic_energies_dict, dict)
            except Exception as e:
                raise RuntimeError(f'E0s specified invalidly, error {e} occured') from e
    else:
        raise RuntimeError(
            'E0s not found in training file and not specified in command line'
        )
    return atomic_energies_dict


def compute_average_atomic_energies(
    dataset,
    z_table: AtomicNumberTable,
    max_n: int | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> dict[int, float]:
    """
    Estimate average interaction energy per chemical element by solving a least
    squares system over configurations in the dataset.
    """
    if max_n is None:
        sample_size = len(dataset)
    else:
        sample_size = min(len(dataset), max_n)

    if sample_size == 0:
        raise ValueError('Cannot compute atomic energies from an empty dataset.')

    if rng is None:
        rng = np.random.default_rng()

    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    indices = indices[:sample_size]

    len_zs = len(z_table)
    A = np.zeros((sample_size, len_zs), dtype=np.float64)
    B = np.zeros(sample_size, dtype=np.float64)

    for row, idx in enumerate(indices):
        atoms = dataset[idx]
        numbers = atoms.get_atomic_numbers()
        B[row] = atoms.get_potential_energy()
        for col, z in enumerate(z_table):
            A[row, col] = np.count_nonzero(numbers == z)

    try:
        E0s = np.linalg.lstsq(A, B, rcond=None)[0]
        atomic_energies_dict = {z: float(E0s[i]) for i, z in enumerate(z_table)}
    except np.linalg.LinAlgError:
        logging.warning(
            'Failed to compute E0s using least squares regression, using zeros instead'
        )
        atomic_energies_dict = {z: 0.0 for z in z_table}

    return atomic_energies_dict


__all__ = ['Statistics', 'get_atomic_energies', 'compute_average_atomic_energies']
