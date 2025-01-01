import ast
import json
import torch_geometric

from equitrain.data.format_hdf5.dataset import HDF5GraphDataset
from equitrain.data.statistics_data     import Statistics


def get_dataloader(data_file, args, statistics = None, logger=None):

    if data_file is None:
        return None

    if statistics is None:

        statistics = Statistics.load(args.statistics_file)

        if logger is not None:
            logger.log(1, f'Using r_max={statistics.r_max} from statistics file `{args.statistics_file}`')

    data_set = HDF5GraphDataset(
        data_file, r_max=statistics.r_max, atomic_numbers=statistics.atomic_numbers
    )
    data_loader = torch_geometric.loader.DataLoader(
        dataset     = data_set,
        batch_size  = args.batch_size,
        shuffle     = args.shuffle,
        drop_last   = False,
        pin_memory  = args.pin_memory,
        num_workers = args.workers,
    )

    return data_loader


def get_dataloaders(args, logger=None):

    statistics = Statistics.load(args.statistics_file)

    if logger is not None:
        logger.log(1, f'Using r_max={statistics.r_max} from statistics file `{args.statistics_file}`')

    train_loader = get_dataloader(args.train_file, args, statistics = statistics, logger=logger)
    valid_loader = get_dataloader(args.valid_file, args, statistics = statistics, logger=logger)
    test_loader  = get_dataloader(args.test_file , args, statistics = statistics, logger=logger)

    return train_loader, valid_loader, test_loader
