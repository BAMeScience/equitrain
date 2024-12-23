import argparse
import bz2
import glob
import json
import numpy as np
import random
import sys

from pathlib import Path
from tqdm.auto import tqdm

from ase import Atoms
from ase.io import write, read
from ase.units import GPa

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


class WeightedRandomizer:
    def __init__ (self, weights, seed=42):
        random.seed(seed)
        self.__max = .0
        self.__weights = []
        for value, weight in weights.items ():
            self.__max += weight
            self.__weights.append ( (self.__max, value) )

    def __call__(self):
        r = random.random () * self.__max
        for ceil, value in self.__weights:
            if ceil > r: return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir" , type=str, required=True)
    parser.add_argument("--dst_prefix", type=str, required=True)
    return parser.parse_args()


def main(src_dir: Path | str, dst_prefix: Path | str):

    filename_train = f"{dst_prefix}_train.xyz"
    filename_valid = f"{dst_prefix}_valid.xyz"
    filename_test  = f"{dst_prefix}_test.xyz"

    if Path(filename_train).exists():
        print("Output train file exists. Exiting...")
        sys.exit(1)

    if Path(filename_valid).exists():
        print("Output valid file exists. Exiting...")
        sys.exit(1)

    if Path(filename_test).exists():
        print("Output test file exists. Exiting...")
        sys.exit(1)

    # Random assignment to train, valid, or test set
    rand = WeightedRandomizer({'train': 0.95, 'valid': 0.05, 'test': 0.00})

    # A dictionary used to put all structures with the same composition
    # to go into the same train, valid, or test set
    selection_dict = {}

    r_train = []
    r_valid = []
    r_test  = []

    for src_file in tqdm(glob.glob(f"{src_dir}/*.xyz"), desc="Joining files"):

        selection = rand()

        atoms_list = read(src_file, index=":")

        # Use only three points per trajectory
        if len(atoms_list) > 3:
            i1 = 0
            i2 = round((len(atoms_list)/2)) - 1
            i3 = len(atoms_list) - 1

            atoms_list = [atoms_list[i1], atoms_list[i2], atoms_list[i3]]

        if selection == "train":
            write(filename_train, atoms_list, append=True)
        if selection == "valid":
            write(filename_valid, atoms_list, append=True)
        if selection == "test":
            write(filename_test, atoms_list, append=True)


if __name__ == "__main__":
    args = parse_args()
    main(args.src_dir, args.dst_prefix)
