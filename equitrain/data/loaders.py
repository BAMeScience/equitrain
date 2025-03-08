from pathlib import Path

import torch
from accelerate import Accelerator

from equitrain.data.format_hdf5.dataset import HDF5GraphDataset
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
    args,
    data_file: Path | str,
    atomic_numbers: list[int],
    r_max: float,
    accelerator: Accelerator = None,
):
    if data_file is None:
        return None

    data_set = HDF5GraphDataset(data_file, r_max=r_max, atomic_numbers=atomic_numbers)

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


def get_dataloaders(
    args,
    atomic_numbers: list[int],
    r_max: float,
    accelerator: Accelerator = None,
):
    train_loader = get_dataloader(
        args,
        args.train_file,
        atomic_numbers,
        r_max,
        accelerator,
    )
    valid_loader = get_dataloader(
        args,
        args.valid_file,
        atomic_numbers,
        r_max,
        accelerator,
    )
    test_loader = get_dataloader(
        args,
        args.test_file,
        atomic_numbers,
        r_max,
        accelerator,
    )

    return train_loader, valid_loader, test_loader
