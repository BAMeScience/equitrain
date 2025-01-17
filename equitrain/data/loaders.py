import torch
from accelerate import Accelerator

from equitrain.data.format_hdf5.dataset import HDF5GraphDataset
from equitrain.data.statistics_data import Statistics
from equitrain.logger import FileLogger

from .loaders_dynamic import DynamicGraphLoader


def dataloader_update_errors(
    args,
    dataloader,
    errors: torch.Tensor,
    accelerator: Accelerator = None,
    logger: FileLogger = None,
):
    generator = torch.Generator(device=accelerator.device)

    dataloader = DynamicGraphLoader(
        dataset=dataloader.dataset,
        errors=errors,
        errors_threshold=args.weighted_sampler_threshold,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=args.pin_memory,
        num_workers=args.workers,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
        drop=args.batch_drop,
        generator=generator,
    )

    if accelerator is not None:
        dataloader = accelerator.prepare(dataloader)

    if logger is not None:
        logger.log(1, 'Updating training loader with new sampling weights')

    return dataloader


def get_dataloader(
    data_file,
    args,
    statistics=None,
    accelerator: Accelerator = None,
    logger: FileLogger = None,
):
    if data_file is None:
        return None

    if statistics is None:
        statistics = Statistics.load(args.statistics_file)

        if logger is not None:
            logger.log(
                1,
                f'Using r_max={statistics.r_max} from statistics file `{args.statistics_file}`',
            )

    data_set = HDF5GraphDataset(
        data_file, r_max=statistics.r_max, atomic_numbers=statistics.atomic_numbers
    )

    data_loader = DynamicGraphLoader(
        dataset=data_set,
        errors=None,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        drop_last=False,
        pin_memory=args.pin_memory,
        num_workers=args.workers,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
        drop=args.batch_drop,
    )

    if accelerator is not None:
        data_loader = accelerator.prepare(data_loader)

    return data_loader


def get_dataloaders(args, accelerator: Accelerator = None, logger: FileLogger = None):
    statistics = Statistics.load(args.statistics_file)

    if logger is not None:
        logger.log(
            1,
            f'Using r_max={statistics.r_max} from statistics file `{args.statistics_file}`',
        )

    train_loader = get_dataloader(
        args.train_file,
        args,
        statistics=statistics,
        accelerator=accelerator,
        logger=logger,
    )
    valid_loader = get_dataloader(
        args.valid_file,
        args,
        statistics=statistics,
        accelerator=accelerator,
        logger=logger,
    )
    test_loader = get_dataloader(
        args.test_file,
        args,
        statistics=statistics,
        accelerator=accelerator,
        logger=logger,
    )

    return train_loader, valid_loader, test_loader
