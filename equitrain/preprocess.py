import logging
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch_geometric

from equitrain.argparser import ArgumentError, check_args_complete
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_torch import statistics as torch_statistics
from equitrain.data.configuration import niggli_reduce_inplace
from equitrain.data.format_hdf5 import HDF5Dataset, HDF5GraphDataset
from equitrain.data.format_lmdb import convert_lmdb_to_hdf5
from equitrain.data.format_xyz import XYZReader
from equitrain.data.statistics_data import Statistics, get_atomic_energies
from equitrain.logger import FileLogger


def _collect_atomic_numbers(filename_hdf5: Path) -> AtomicNumberTable | None:
    with HDF5Dataset(filename_hdf5, 'r') as dataset:
        if len(dataset) == 0:
            return None
        numbers: set[int] = set()
        for idx in range(len(dataset)):
            numbers.update(int(z) for z in dataset[idx].get_atomic_numbers())
    return AtomicNumberTable(sorted(numbers))


def _convert_to_hdf5(
    args,
    source_filename,
    filename_hdf5,
    extract_atomic_numbers=False,
    extract_atomic_energies=False,
    *,
    niggli_reduce=False,
):
    atomic_numbers = None
    atomic_energies = None

    source_path = Path(source_filename)
    target_path = Path(filename_hdf5)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    atoms_transform = niggli_reduce_inplace if niggli_reduce else None

    lower_name = source_path.name.lower()
    if lower_name.endswith('.xyz') or lower_name.endswith('.xyz.gz'):
        reader = XYZReader(
            filename=source_path,
            energy_key=args.energy_key,
            forces_key=args.forces_key,
            stress_key=args.stress_key,
            extract_atomic_numbers=extract_atomic_numbers,
            extract_atomic_energies=extract_atomic_energies,
        )
        with HDF5Dataset(target_path, 'w') as file:
            for i, atoms in enumerate(reader):
                if atoms_transform is not None:
                    atoms_transform(atoms)
                file[i] = atoms
        if extract_atomic_numbers:
            atomic_numbers = reader.atomic_numbers
        if extract_atomic_energies:
            atomic_energies = reader.atomic_energies

    elif lower_name.endswith('.lmdb') or lower_name.endswith('.aselmdb'):
        convert_lmdb_to_hdf5(
            source_path,
            target_path,
            atoms_transform=atoms_transform,
            overwrite=True,
            show_progress=False,
        )
        if extract_atomic_numbers:
            atomic_numbers = _collect_atomic_numbers(target_path)

    elif lower_name.endswith('.h5') or lower_name.endswith('.hdf5'):
        if source_path.resolve() != target_path.resolve():
            shutil.copyfile(source_path, target_path)
        if niggli_reduce:
            logging.warning(
                'Requested Niggli reduction is skipped for existing HDF5 source %s. '
                'Regenerate the dataset from raw structures if reduction is required.',
                source_path,
            )
        if extract_atomic_numbers:
            atomic_numbers = _collect_atomic_numbers(target_path)

    else:
        raise ArgumentError(f'Unsupported dataset format: {source_filename}')

    return atomic_numbers, atomic_energies


def _preprocess(args):
    """
    This script loads an xyz dataset and prepares
    new hdf5 file that is ready for training with on-the-fly dataloading
    """
    logger = FileLogger(
        log_to_file=False, enable_logging=True, output_dir=None, verbosity=args.verbose
    )

    backend_name = getattr(args, 'backend', 'torch') or 'torch'

    if backend_name == 'torch':
        from equitrain.backends.torch_utils import (
            set_dtype as torch_set_dtype,
        )
        from equitrain.backends.torch_utils import (
            set_seeds as torch_set_seeds,
        )

        torch_set_seeds(args.seed)
        torch_set_dtype(args.dtype)
    elif backend_name == 'jax':
        from equitrain.backends.jax_utils import set_jax_dtype

        np.random.seed(args.seed)
        random.seed(args.seed)
        set_jax_dtype(args.dtype)
    else:
        raise ArgumentError(f'Unsupported backend: {backend_name}')

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
            atomic_numbers, atomic_energies = _convert_to_hdf5(
                args,
                args.train_file,
                filename_train,
                extract_atomic_numbers=(
                    args.compute_statistics and statistics.atomic_numbers is None
                ),
                extract_atomic_energies=(
                    args.compute_statistics and statistics.atomic_energies is None
                ),
                niggli_reduce=args.niggli_reduce,
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
            _convert_to_hdf5(
                args,
                args.valid_file,
                filename_valid,
                niggli_reduce=args.niggli_reduce,
            )

    # Convert test file
    if args.test_file:
        if Path(filename_test).exists():
            logger.log(1, 'Test file exists. Skipping...')

        else:
            logger.log(1, 'Converting test file')
            _convert_to_hdf5(
                args,
                args.test_file,
                filename_test,
                niggli_reduce=args.niggli_reduce,
            )

    if Path(filename_train).exists() and args.compute_statistics:
        logger.log(1, 'Computing statistics')

        # Compute statistics
        with HDF5Dataset(filename_train) as train_dataset:
            # If training set did not contain any single atom entries, estimate E0s...
            if statistics.atomic_numbers is None or len(statistics.atomic_numbers) == 0:
                statistics.atomic_numbers = torch_statistics.compute_atomic_numbers(
                    train_dataset
                )

            # If training set did not contain any single atom entries, estimate E0s...
            if (
                statistics.atomic_energies is None
                or len(statistics.atomic_energies) == 0
            ):
                statistics.atomic_energies = get_atomic_energies(
                    args.atomic_energies, train_dataset, statistics.atomic_numbers
                )

        if getattr(args, 'backend', 'torch') == 'jax':
            from equitrain.data.backend_jax import atoms_to_graphs, build_loader
            from equitrain.data.backend_jax import statistics as jax_statistics

            if statistics.r_max is None:
                raise RuntimeError(
                    'JAX preprocessing requires --r-max to be specified.'
                )

            jax_z_table = AtomicNumberTable(list(statistics.atomic_numbers))
            jax_graphs = atoms_to_graphs(
                filename_train,
                statistics.r_max,
                jax_z_table,
                niggli_reduce=args.niggli_reduce,
            )
            if not jax_graphs:
                raise RuntimeError('Training dataset is empty.')

            jax_loader = build_loader(
                jax_graphs,
                batch_size=args.batch_size,
                shuffle=False,
                max_nodes=args.batch_max_nodes,
                max_edges=args.batch_max_edges,
            )

            statistics.avg_num_neighbors, statistics.mean, statistics.std = (
                jax_statistics.compute_statistics(
                    jax_loader,
                    statistics.atomic_energies,
                    statistics.atomic_numbers,
                )
            )

        else:
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
                (
                    statistics.avg_num_neighbors,
                    statistics.mean,
                    statistics.std,
                ) = torch_statistics.compute_statistics(
                    train_loader,
                    statistics.atomic_energies,
                    statistics.atomic_numbers,
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


import sys as _sys

if 'equitrain' in _sys.modules:
    setattr(_sys.modules['equitrain'], 'preprocess', preprocess)
