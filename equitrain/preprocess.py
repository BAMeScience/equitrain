import os
from pathlib import Path

import torch_geometric

from equitrain.argparser import ArgumentError, check_args_complete
from equitrain.data import (
    AtomicNumberTable,
    Statistics,
    compute_atomic_numbers,
    compute_statistics,
    get_atomic_energies,
)
from equitrain.data.format_hdf5 import HDF5Dataset, HDF5GraphDataset
from equitrain.data.format_xyz import XYZReader
from equitrain.logger import FileLogger
from equitrain.utility import set_dtype, set_seeds


def _convert_xyz_to_hdf5(
    args,
    filename_xyz,
    filename_hdf5,
    extract_atomic_numbers=False,
    extract_atomic_energies=False,
):
    atomic_numbers = None
    atomic_energies = None

    reader = XYZReader(
        filename=filename_xyz,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        stress_key=args.stress_key,
        extract_atomic_numbers=extract_atomic_numbers,
        extract_atomic_energies=extract_atomic_energies,
    )

    # Open HDF5 file in write mode
    with HDF5Dataset(filename_hdf5, 'w') as file:
        for i, config in enumerate(reader):
            file[i] = config

    if extract_atomic_numbers:
        atomic_numbers = reader.atomic_numbers

    if extract_atomic_energies:
        atomic_energies = reader.atomic_energies

    return atomic_numbers, atomic_energies


def _preprocess(args):
    """
    This script loads an xyz dataset and prepares
    new hdf5 file that is ready for training with on-the-fly dataloading
    """
    logger = FileLogger(
        log_to_file=False, enable_logging=True, output_dir=None, verbosity=args.verbose
    )

    set_seeds(args.seed)
    set_dtype(args.dtype)

    filename_train = os.path.join(args.output_dir, 'train.h5')
    filename_valid = os.path.join(args.output_dir, 'valid.h5')
    filename_test = os.path.join(args.output_dir, 'test.h5')

    statistics = Statistics(r_max=args.r_max)

    # Read atomic numbers from arguments if available
    if args.atomic_numbers is not None:
        logger.log(1, 'Using atomic numbers from command line argument')
        statistics.atomic_numbers = AtomicNumberTable.from_str(args.atomic_numbers)

    # Convert training file and obtain z_table and atomit_energies if required
    if args.train_file:
        if Path(filename_train).exists():
            logger.log(1, 'Train file exists. Skipping...')

        else:
            logger.log(1, 'Converting train file')
            atomic_numbers, atomic_energies = _convert_xyz_to_hdf5(
                args,
                args.train_file,
                filename_train,
                extract_atomic_numbers=(
                    args.compute_statistics and statistics.atomic_numbers is None
                ),
                extract_atomic_energies=(
                    args.compute_statistics and statistics.atomic_energies is None
                ),
            )

            if statistics.atomic_numbers is None:
                statistics.atomic_numbers = atomic_numbers

            if statistics.atomic_energies is None:
                statistics.atomic_energies = atomic_energies

    # Convert validation file
    if args.valid_file:
        if Path(filename_valid).exists():
            logger.log(1, 'Validation file exists. Skipping...')

        else:
            logger.log(1, 'Converting valid file')
            _convert_xyz_to_hdf5(args, args.valid_file, filename_valid)

    # Convert test file
    if args.test_file:
        if Path(filename_test).exists():
            logger.log(1, 'Test file exists. Skipping...')

        else:
            logger.log(1, 'Converting test file')
            _convert_xyz_to_hdf5(args, args.test_file, filename_test)

    if Path(filename_train).exists() and args.compute_statistics:
        logger.log(1, 'Computing statistics')

        # Compute statistics
        with HDF5Dataset(filename_train) as train_dataset:
            # If training set did not contain any single atom entries, estimate E0s...
            if statistics.atomic_numbers is None or len(statistics.atomic_numbers) == 0:
                statistics.atomic_numbers = compute_atomic_numbers(train_dataset)

            # If training set did not contain any single atom entries, estimate E0s...
            if (
                statistics.atomic_energies is None
                or len(statistics.atomic_energies) == 0
            ):
                statistics.atomic_energies = get_atomic_energies(
                    args.atomic_energies, train_dataset, statistics.atomic_numbers
                )

        with HDF5GraphDataset(
            filename_train,
            r_max=statistics.r_max,
            atomic_numbers=statistics.atomic_numbers,
        ) as train_dataset:
            train_loader = torch_geometric.loader.DataLoader(
                dataset=train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
            )
            statistics.avg_num_neighbors, statistics.mean, statistics.std = (
                compute_statistics(
                    train_loader,
                    statistics.atomic_energies,
                    statistics.atomic_numbers,
                )
            )

            logger.log(1, f'Final statistics to be saved: {statistics}')

            statistics.dump(os.path.join(args.output_dir, 'statistics.json'))


def preprocess(args):
    check_args_complete(args, 'preprocess')

    if args.train_file is None:
        raise ArgumentError('--train-file is a required argument')
    if args.output_dir is None:
        raise ArgumentError('--output-dir is a required argument')

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    _preprocess(args)
