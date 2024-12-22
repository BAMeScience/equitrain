import ast
import json
import torch_geometric

from equitrain.data.format_hdf5.dataset import HDF5Dataset
from equitrain.data.statistics          import get_atomic_number_table_from_zs


def get_dataloader(data_file, args, shuffle=False, logger=None):
    with open(args.statistics_file, "r") as f:
        statistics = json.load(f)

    zs_list = ast.literal_eval(statistics["atomic_numbers"])
    z_table = get_atomic_number_table_from_zs(zs_list)
    r_max = float(statistics["r_max"])

    if logger is not None:
        logger.info(f'Using r_max={r_max} from statistics file `{args.statistics_file}`')

    data_set = HDF5Dataset(
        data_file, r_max=r_max, z_table=z_table
    )
    data_loader = torch_geometric.loader.DataLoader(
        dataset=data_set,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=args.pin_mem,
        num_workers=args.workers,
    )

    return data_loader, r_max


def get_dataloaders(args, logger=None):

    with open(args.statistics_file, "r") as f:
        statistics = json.load(f)

    zs_list = ast.literal_eval(statistics["atomic_numbers"])
    z_table = get_atomic_number_table_from_zs(zs_list)
    r_max = float(statistics["r_max"])

    if logger is not None:
        logger.info(f'Using r_max={r_max} from statistics file `{args.statistics_file}`')

    if args.train_file is None:
        train_loader = None
    else:
        train_set = HDF5Dataset(
            args.train_file, r_max=r_max, z_table=z_table
        )
        train_loader = torch_geometric.loader.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            drop_last=False,
            pin_memory=args.pin_mem,
            num_workers=args.workers,
        )

    if args.valid_file is None:
        valid_loader = None
    else:
        valid_set = HDF5Dataset(
            args.valid_file, r_max=r_max, z_table=z_table
        )
        valid_loader = torch_geometric.loader.DataLoader(
            dataset=valid_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=args.pin_mem,
            num_workers=args.workers,
        )

    if args.test_file is None:
        test_loader = None
    else:
        test_set = HDF5Dataset(
            args.test_file, r_max=r_max, z_table=z_table
        )
        test_loader = torch_geometric.loader.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=args.pin_mem,
            num_workers=args.workers,
        )

    return train_loader, valid_loader, test_loader, r_max
