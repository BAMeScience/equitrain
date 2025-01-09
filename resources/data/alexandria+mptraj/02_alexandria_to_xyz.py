import argparse
import bz2
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import write
from ase.units import GPa
from filelock import FileLock
from pqdm.processes import pqdm
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True)
    parser.add_argument('--dst_dir', type=str, required=True)
    parser.add_argument('--n-jobs', type=int, default=1)
    return parser.parse_args()


def convert_json_item_to_atoms(alex_id, calc_id, ionic_step, w):
    atoms = AseAtomsAdaptor.get_atoms(Structure.from_dict(w['structure']))
    atoms.arrays['forces'] = np.array(w['forces'])
    atoms.info = {
        'config_type': 'Default',
        'alex_id': alex_id,
        'calc_id': calc_id,
        'ionic_step': ionic_step,
        'energy': w['energy'],
        'stress': -np.array(w['stress']).flatten() * 1e-1 * GPa,
    }
    return atoms


def convert_alexandria_to_atoms(src_path: Path | str):
    """Extracts the structures from a bz2 compressed json file and writes them to an extended xyz file."""

    if isinstance(src_path, str):
        src_path = Path(src_path)

    with bz2.open(src_path, 'rb') as f:
        data = json.load(f)

    assert isinstance(data, dict)

    result = dict()

    for alex_id, u in data.items():
        for calc_id, v in enumerate(u):
            for ionic_step, w in enumerate(v['steps']):
                try:
                    atoms = convert_json_item_to_atoms(alex_id, calc_id, ionic_step, w)

                    trajectory = result.get(atoms.get_chemical_formula(), [])
                    trajectory.append(atoms)

                    result[atoms.get_chemical_formula()] = trajectory

                except ValueError:
                    # Catch one entry that seems to have invalid forces array
                    pass

    return result


def export_result(result, dst_dir):
    if isinstance(dst_dir, str):
        dst_dir = Path(dst_dir)

    dst_dir.mkdir(exist_ok=True, parents=True)

    for formula, trajectory in result.items():
        filename = dst_dir / f'{formula}.xyz'
        filename_lock = dst_dir / f'{formula}.lock'

        with FileLock(filename_lock):
            write(filename, trajectory, append=True)


def convert_file(src_path, dst_dir):
    result = convert_alexandria_to_atoms(src_path)
    export_result(result, dst_dir)


def main(src_dir: Path | str, dst_dir: Path | str, n_jobs: int):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not src_dir.is_dir():
        print(f"Source directory '{src_dir}' does not exist or is not a directory.")
        sys.exit(1)

    if dst_dir.exists():
        print(f"Output path '{dst_dir}' exists. Exiting to avoid duplicated entries...")
        sys.exit(1)

    files = list(src_dir.rglob('*.json.bz2'))
    args = [(str(src_file), str(dst_dir)) for src_file in files]

    pqdm(args, convert_file, argument_type='args', n_jobs=n_jobs)


if __name__ == '__main__':
    args = parse_args()
    main(args.src_dir, args.dst_dir, args.n_jobs)
