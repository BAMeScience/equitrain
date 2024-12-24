# This file loads an xyz dataset and prepares
# new hdf5 file that is ready for training with on-the-fly dataloading

import logging
import ast
import numpy as np
import json
import random
import torch_geometric
import os

from pathlib import Path

from equitrain.argparser import ArgumentError
from equitrain.data import SubsetCollection, compute_statistics, get_atomic_energies, get_atomic_number_table_from_zs
from equitrain.data.format_hdf5 import HDF5Dataset, HDF5GraphDataset
from equitrain.data.format_xyz import XYZReader
from equitrain.utility import set_dtype, set_seeds


def _convert_xyz_to_hdf5(args, filename_xyz, filename_hdf5, extract_z_table = False, extract_atomic_energies = False):

    try:
        config_type_weights = ast.literal_eval(args.config_type_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}

    z_table = None
    atomic_energies_dict = None

    reader = XYZReader(
        filename                = filename_xyz,
        config_type_weights     = config_type_weights,
        energy_key              = args.energy_key,
        forces_key              = args.forces_key,
        stress_key              = args.stress_key,
        extract_atomic_energies = True,
    )

    # Open HDF5 file in write mode
    with HDF5Dataset(filename_hdf5, "w") as file:

        for i, config in enumerate(reader):
            file.save_configuration(config, i)

    if extract_z_table:
        z_table = reader.z_table

    if extract_atomic_energies:
        atomic_energies_dict = reader.atomic_energies_dict

    return z_table, atomic_energies_dict


def _preprocess(args):
    """
    This script loads an xyz dataset and prepares
    new hdf5 file that is ready for training with on-the-fly dataloading
    """

    set_seeds(args.seed)
    set_dtype(args.dtype)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )

    filename_train = os.path.join(args.output_dir, "train.h5")
    filename_valid = os.path.join(args.output_dir, "valid.h5")
    filename_test  = os.path.join(args.output_dir, "test.h5")

    z_table = None
    atomic_energies_dict = None

    # Read atomic numbers from arguments if available
    if args.atomic_numbers is not None:
        logging.info("Using atomic numbers from command line argument")
        zs_list = ast.literal_eval(args.atomic_numbers)
        assert isinstance(zs_list, list)
        z_table = get_atomic_number_table_from_zs(zs_list)

    # Convert training file and obtain z_table and atomit_energies if required
    if args.train_file:
        logging.info("Converting train file")
        _z_table, _atomic_energies_dict = _convert_xyz_to_hdf5(
            args,
            args.train_file,
            filename_train,
            extract_z_table         = (args.compute_statistics and z_table              is None),
            extract_atomic_energies = (args.compute_statistics and atomic_energies_dict is None),
            )

        if z_table is None:
            z_table = _z_table

        if atomic_energies_dict is None:
            atomic_energies_dict = _atomic_energies_dict

    # Convert validation file
    if args.valid_file:
        logging.info("Converting valid file")
        _convert_xyz_to_hdf5(args, args.valid_file, filename_valid)

    # Convert test file
    if args.test_file:
        logging.info("Converting test file")
        _convert_xyz_to_hdf5(args, args.test_file, filename_test)

    if args.train_file and args.compute_statistics:
        # Compute statistics
        logging.info("Computing statistics")
        # Remove all items from z_table for which no atomic energies exits
        z_table.zs = sorted(list(set(z_table.zs).intersection(atomic_energies_dict.keys())))
        # Sort atomic energies according to z_table
        atomic_energies: np.ndarray = np.array(
            [atomic_energies_dict[z] for z in z_table.zs]
        )
        logging.info(f"Atomic energies array for computation: {atomic_energies.tolist()}")

        with HDF5GraphDataset(filename_train, r_max=args.r_max, z_table=z_table) as train_dataset:

            train_loader = torch_geometric.loader.DataLoader(
                dataset    = train_dataset,
                batch_size = args.batch_size,
                shuffle    = False,
                drop_last  = False,
            )
            avg_num_neighbors, mean, std = compute_statistics(
                train_loader, atomic_energies, z_table
            )
            logging.info(f"Average number of neighbors: {avg_num_neighbors}")
            logging.info(f"Mean                       : {mean}")
            logging.info(f"Standard deviation         : {std}")

            # save the statistics as a json
            statistics = {
                "atomic_energies"  : { int(k): float(v) for k, v in atomic_energies_dict.items() },
                "avg_num_neighbors": avg_num_neighbors,
                "mean"             : mean,
                "std"              : std,
                "atomic_numbers"   : z_table.zs,
                "r_max"            : args.r_max,
            }
            logging.info(f"Final statistics to be saved: {statistics}")

            with open(os.path.join(args.output_dir, "statistics.json"), "w") as f:
                json.dump(statistics, f)


def preprocess(args):
    if args.train_file is None:
        raise ArgumentError("--train-file is a required argument")
    if args.valid_file is None:
        raise ArgumentError("--valid-file is a required argument")
    if args.statistics_file is None:
        raise ArgumentError("--statistics-file is a required argument")
    if args.output_dir is None:
        raise ArgumentError("--output-dir is a required argument")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    _preprocess(args)
