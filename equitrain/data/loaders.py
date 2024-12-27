import ast
import json
import torch_geometric

from equitrain.data.format_hdf5.dataset import HDF5GraphDataset
from equitrain.data.statistics_data     import Statistics


def get_dataloader(data_file, args, shuffle=False, logger=None):

    statistics = Statistics.load(args.statistics_file)

    if logger is not None:
        logger.info(f'Using r_max={statistics.r_max} from statistics file `{args.statistics_file}`')

    data_set = HDF5GraphDataset(
        data_file, r_max=statistics.r_max, z_table=statistic.atomic_numbers
    )
    data_loader = torch_geometric.loader.DataLoader(
        dataset     = data_set,
        batch_size  = args.batch_size,
        shuffle     = shuffle,
        drop_last   = False,
        pin_memory  = args.pin_mem,
        num_workers = args.workers,
    )

    return data_loader


def get_dataloaders(args, logger=None):

    statistics = Statistics.load(args.statistics_file)

    if logger is not None:
        logger.info(f'Using r_max={statistics.r_max} from statistics file `{args.statistics_file}`')

    if args.train_file is None:
        train_loader = None
    else:
        train_set = HDF5GraphDataset(
            args.train_file, r_max=statistics.r_max, z_table=statistics.atomic_numbers
        )
        train_loader = torch_geometric.loader.DataLoader(
            dataset     = train_set,
            batch_size  = args.batch_size,
            shuffle     = args.shuffle,
            drop_last   = False,
            pin_memory  = args.pin_mem,
            num_workers = args.workers,
        )

    if args.valid_file is None:
        valid_loader = None
    else:
        valid_set = HDF5GraphDataset(
            args.valid_file, r_max=statistics.r_max, z_table=statistics.atomic_numbers
        )
        valid_loader = torch_geometric.loader.DataLoader(
            dataset     = valid_set,
            batch_size  = args.batch_size,
            shuffle     = False,
            drop_last   = False,
            pin_memory  = args.pin_mem,
            num_workers = args.workers,
        )

    if args.test_file is None:
        test_loader = None
    else:
        test_set = HDF5GraphDataset(
            args.test_file, r_max=statistics.r_max, z_table=statistics.atomic_numbers
        )
        test_loader = torch_geometric.loader.DataLoader(
            dataset     = test_set,
            batch_size  = args.batch_size,
            shuffle     = False,
            drop_last   = False,
            pin_memory  = args.pin_mem,
            num_workers = args.workers,
        )

    return train_loader, valid_loader, test_loader
