import argparse
import glob
import random
import sys

from pathlib import Path
from filelock import FileLock
from pqdm.processes import pqdm
from ase.io import write, read


class WeightedRandomizer:
    def __init__ (self, weights, seed=42):
        random.seed(seed)
        self.__max     = 0.0
        self.__weights = []
        for value, weight in weights.items ():
            self.__max += weight
            self.__weights.append((self.__max, value))

    def __call__(self):
        r = random.random () * self.__max
        for ceil, value in self.__weights:
            if ceil > r: return value


def batch_files(file_list, batch_size):
    return [file_list[i:i + batch_size] for i in range(0, len(file_list), batch_size)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir"     , type=str  , required=True)
    parser.add_argument("--dst_prefix"  , type=str  , required=True)
    parser.add_argument("--train_weight", type=float, default=0.95)
    parser.add_argument("--valid_weight", type=float, default=0.05)
    parser.add_argument("--test_weight" , type=float, default=0.00)
    parser.add_argument("--batch_size"  , type=int  , default=1000)
    parser.add_argument("--n_jobs"      , type=int  , default=1)
    return parser.parse_args()


def process_batch(src_files, filename_train, filename_valid, filename_test, rand):

    result_train = []
    result_valid = []
    result_test  = []

    for src_file in src_files:

        selection = rand()

        atoms_list = read(src_file, index=":")

        # Use only three points per trajectory
        if len(atoms_list) > 3:
            i1 = 0
            i2 = round((len(atoms_list)/2)) - 1
            i3 = len(atoms_list) - 1

            atoms_list = [atoms_list[i1], atoms_list[i2], atoms_list[i3]]

        if selection == "train":
            result_train.extend(atoms_list)
        if selection == "valid":
            result_valid.extend(atoms_list)
        if selection == "test":
            result_test.extend(atoms_list)
    
    # Export train data
    if len(result_train) > 0:
        with FileLock(filename_train.with_suffix(".lock")):
            write(filename_train, result_train, append=True)

    # Export train data
    if len(result_valid) > 0:
        with FileLock(filename_valid.with_suffix(".lock")):
            write(filename_valid, result_valid, append=True)
  
    # Export test data
    if len(result_test) > 0:
        with FileLock(filename_test.with_suffix(".lock")):
            write(filename_test, result_test, append=True)


def main(
    src_dir     : Path | str,
    dst_prefix  : Path | str,
    train_weight: float,
    valid_weight: float,
    test_weight : float,
    batch_size  : int,
    n_jobs      : int,
    ):

    filename_train = Path(f"{dst_prefix}_train.xyz")
    filename_valid = Path(f"{dst_prefix}_valid.xyz")
    filename_test  = Path(f"{dst_prefix}_test.xyz")

    if filename_train.exists():
        print("Output train file exists. Exiting...")
        sys.exit(1)

    if filename_valid.exists():
        print("Output valid file exists. Exiting...")
        sys.exit(1)

    if filename_test.exists():
        print("Output test file exists. Exiting...")
        sys.exit(1)

    # Random assignment to train, valid, or test set
    rand = WeightedRandomizer({'train': train_weight, 'valid': valid_weight, 'test': test_weight})

    batches = batch_files(glob.glob(f"{src_dir}/*.xyz"), batch_size)
    args    = [(batch, filename_train, filename_valid, filename_test, rand) for batch in batches]

    pqdm(args, process_batch, argument_type="args", n_jobs=n_jobs)


if __name__ == "__main__":

    args = parse_args()

    main(args.src_dir, args.dst_prefix, args.train_weight, args.valid_weight, args.test_weight, args.batch_size, args.n_jobs)
