"""Optional adapter for the separately installed RTorch HDF5 graph loader."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch


def _import_rtorch() -> Any:
    try:
        import rtorch
    except ImportError as error:
        raise RuntimeError(
            "--torch-loader rtorch requires the separately installed RTorch extension. "
            "Build/install RTorch first, then rerun Equitrain with --torch-loader rtorch."
        ) from error
    return rtorch


class RTorchGraphLoader:
    """Adapt RTorch's native loader to Equitrain's list-of-PyG-batches contract."""

    _uses_rtorch = True

    def __init__(self, native_loader: Any, device: torch.device) -> None:
        self._native_loader = native_loader
        self.dataset = native_loader.dataset
        self._device = device

    def __len__(self) -> int:
        return len(self._native_loader)

    def __iter__(self) -> Iterator[list[Any]]:
        non_blocking = self._device.type == 'cuda'
        for batches in self._native_loader:
            yield [batch.to(self._device, non_blocking=non_blocking) for batch in batches]


def get_rtorch_dataloader(
    *,
    files: list[Path],
    atomic_numbers: list[int],
    r_max: float,
    args: Any,
    accelerator: Any = None,
) -> RTorchGraphLoader:
    """Create a single-process native HDF5 graph loader.

    RTorch performs loading/collation natively and does not implement PyTorch
    workers, distributed sharding, reaction grouping, or weighted resampling.
    """
    process_count = int(getattr(accelerator, 'num_processes', 1) or 1)
    if process_count != 1:
        raise RuntimeError(
            '--torch-loader rtorch currently supports single-process training only; '
            'use --torch-loader python for distributed training.'
        )
    rtorch = _import_rtorch()
    device = getattr(accelerator, 'device', torch.device('cpu'))
    native_loader = rtorch.HDF5GraphLoader.from_hdf5_files(
        [str(path) for path in files],
        r_max,
        list(atomic_numbers),
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        drop_last=False,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
        drop=args.batch_drop,
        seed=args.seed,
    )
    return RTorchGraphLoader(native_loader, torch.device(device))
