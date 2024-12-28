
import ast
import json
import logging

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable

from .atomic import AtomicNumberTable
from .statistics import compute_average_atomic_energies


@dataclass
class Statistics:

    atomic_numbers    : AtomicNumberTable = None
    atomic_energies   : Dict[int, float]  = None

    # Energy statistics
    mean              : float             = None
    std               : float             = None

    avg_num_neighbors : float             = None

    r_max             : float             = None

    @classmethod
    def load(cls, filename : Path | str) -> Dict:

        with open(filename, "r") as f:
            statistics_dict = json.load(f)

        statistics = cls(**statistics_dict)
        # Convert atomic numbers list
        statistics.atomic_numbers  = AtomicNumberTable(statistics.atomic_numbers)
        # JSON requires dict keys to be quoted, convert back to integers
        statistics.atomic_energies = { int(key): float(value) for key, value in statistics.atomic_energies.items() }

        return statistics


    def dump(self, filename : Path | str):
        with open(filename, 'w') as f:
            json.dump(asdict(self), f, indent=4)


def get_atomic_energies(E0s, dataset, z_table) -> dict:
    if E0s is not None:
        logging.info(
            "Atomic Energies not in training file, using command line argument E0s"
        )
        if E0s.lower() == "average":
            logging.info(
                "Computing average Atomic Energies using least squares regression"
            )
            # catch if colections.train not defined above
            try:
                assert dataset is not None
                atomic_energies_dict = compute_average_atomic_energies(
                    dataset, z_table
                )
            except Exception as e:
                raise RuntimeError(
                    f"Could not compute average E0s if no training xyz given, error {e} occured"
                ) from e
        else:
            try:
                atomic_energies_dict = ast.literal_eval(E0s)
                assert isinstance(atomic_energies_dict, dict)
            except Exception as e:
                raise RuntimeError(
                    f"E0s specified invalidly, error {e} occured"
                ) from e
    else:
        raise RuntimeError(
            "E0s not found in training file and not specified in command line"
        )
    return atomic_energies_dict
