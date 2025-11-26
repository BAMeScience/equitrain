from __future__ import annotations

from pathlib import Path

import torch
from accelerate import Accelerator

from equitrain.data.format_hdf5.dataset import HDF5GraphDataset
from equitrain.logger import FileLogger

from .loaders_impl import DynamicGraphLoader


def _should_pin_memory(requested: bool, accelerator: Accelerator | None) -> bool:
    if not requested:
        return False
    if accelerator is None:
        return torch.cuda.is_available()
    device = getattr(accelerator, 'device', None)
    return device is not None and getattr(device, 'type', '').startswith('cuda')


def _resolve_num_workers(requested: int, accelerator: Accelerator | None) -> int:
    if requested <= 0:
        return 0
    if accelerator is None:
        return 0
    device = getattr(accelerator, 'device', None)
    if device is None or getattr(device, 'type', '') != 'cuda':
        return 0
    return requested


def dataloader_update_errors(
    args,
    dataloader,
    errors: torch.Tensor,
    accelerator: Accelerator = None,
    logger: FileLogger = None,
):
    pin_memory = _should_pin_memory(args.pin_memory, accelerator)
    num_workers = _resolve_num_workers(args.workers, accelerator)

    if accelerator is not None and getattr(accelerator.device, 'type', '') == 'cuda':
        generator = torch.Generator(device=accelerator.device)
    else:
        generator = torch.Generator()

    dataloader = DynamicGraphLoader(
        dataset=dataloader.dataset,
        errors=errors,
        errors_threshold=args.weighted_sampler_threshold,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
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

    niggli_reduce = getattr(args, 'niggli_reduce', False)
    data_set = HDF5GraphDataset(
        data_file,
        r_max=r_max,
        atomic_numbers=atomic_numbers,
        niggli_reduce=niggli_reduce,
    )

    pin_memory = _should_pin_memory(args.pin_memory, accelerator)
    num_workers = _resolve_num_workers(args.workers, accelerator)

    data_loader = DynamicGraphLoader(
        dataset=data_set,
        errors=None,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
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
