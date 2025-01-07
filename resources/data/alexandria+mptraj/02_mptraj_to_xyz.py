# %% Imports
import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import write
from ase.units import GPa
from filelock import FileLock
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm


# %%
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file', type=str, required=True)
    parser.add_argument('--dst_dir', type=str, required=True)
    parser.add_argument('--n-jobs', type=int, default=1)
    return parser.parse_args()


# %%
def export_result(result, dst_dir: Path | str):
    dst_dir = Path(dst_dir)

    dst_dir.mkdir(exist_ok=True, parents=True)

    for formula, trajectory in result.items():
        filename = dst_dir / f'{formula}.xyz'
        filename_lock = dst_dir / f'{formula}.lock'

        # Do not save result if already in Alexandria dataset
        if not filename.exists():
            with FileLock(filename_lock):
                write(filename, trajectory)


# %%
def main(src_file: Path | str, dst_dir: Path | str, n_jobs: int):
    with open(src_file) as f:
        js = json.load(f)

    result = dict()

    for _, values in tqdm(js.items(), desc='Converting data', total=len(js)):
        # formula = None  # ! unused
        trajectory = []

        # Extract trajectory
        for _, subvalues in values.items():
            atoms = AseAtomsAdaptor.get_atoms(
                Structure.from_dict(subvalues['structure']),
                info={
                    'config_type': 'Default',
                    'energy': subvalues['uncorrected_total_energy'],
                    'stress': -np.array(subvalues['stress']) * 1e-1 * GPa,
                },
            )
            atoms.arrays['forces'] = np.array(subvalues['force'])

            trajectory = result.get(atoms.get_chemical_formula(), [])
            trajectory.append(atoms)

            result[atoms.get_chemical_formula()] = trajectory

    export_result(result, dst_dir)


# %%
if __name__ == '__main__':
    args = parse_args()
    main(args.src_file, args.dst_dir, args.n_jobs)
