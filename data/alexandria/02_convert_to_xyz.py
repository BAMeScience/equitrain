import argparse
import bz2
import glob
import json
import numpy as np
import sys

from pathlib import Path
from tqdm.auto import tqdm

from ase import Atoms
from ase.io import write, read
from ase.units import GPa

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True)
    parser.add_argument("--dst_dir", type=str, required=True)
    return parser.parse_args()


def convert_alexandria_to_atoms(src_path: Path | str):
    """Extracts the structures from a bz2 compressed json file and writes them to an extended xyz file."""

    if isinstance(src_path, str):
        src_path = Path(src_path)

    with bz2.open(src_path, "rb") as f:
        data = json.load(f)

    assert isinstance(data, dict)

    result = dict()

    for alex_id, u in tqdm(data.items(), desc="Converting structures"):
        for calc_id, v in enumerate(u):
            for ionic_step, w in enumerate(v["steps"]):
                atoms = AseAtomsAdaptor.get_atoms(Structure.from_dict(w["structure"]))
                atoms.arrays['forces'] = np.array(w["forces"])
                atoms.info = {
                    'config_type' : 'Default',
                    "alex_id"     : alex_id,
                    "calc_id"     : calc_id,
                    "ionic_step"  : ionic_step,
                    'energy'      : w["energy"],
                    'stress'      : -np.array(w["stress"]).flatten() * 1e-1 * GPa
                }

                trajectory = result.get(atoms.get_chemical_formula(), [])
                trajectory.append(atoms)

                result[atoms.get_chemical_formula()] = trajectory
    
    return result


def export_result(result, dst_dir):

    if isinstance(dst_dir, str):
        dst_dir = Path(dst_dir)

    dst_dir.mkdir(exist_ok=True, parents=True)

    for formula, trajectory in tqdm(result.items(), desc="Exporting trajectories"):
        write(dst_dir / f"{formula}.xyz", trajectory, append=True)


def main(src_dir: Path | str, dst_dir: Path | str):

    if Path(dst_dir).exists():
        print("Output path exists. Exiting...")
        sys.exit(1)

    for src_path in glob.glob(f"{src_dir}/*.json.bz2"):

        print(f"Processing {src_path}...")
        result = convert_alexandria_to_atoms(src_path)
        export_result(result, dst_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args.src_dir, args.dst_dir)
