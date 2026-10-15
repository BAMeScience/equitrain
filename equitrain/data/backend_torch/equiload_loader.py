"""Optional adapter for the separately installed EquiLoad HDF5 graph loader."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch


def _supported_torch_version() -> bool:
    """Whether the installed Torch release is supported by EquiLoad."""
    try:
        major, minor, *_ = torch.__version__.split('+', maxsplit=1)[0].split('.')
        return int(major) == 2 and 6 <= int(minor) < 15
    except (TypeError, ValueError):
        return False


def _import_equiload() -> Any:
    if not _supported_torch_version():
        raise RuntimeError(
            '--torch-loader equiload supports Torch >=2.6,<2.15. '
            f'Found Torch {torch.__version__}; use --torch-loader python instead.'
        )
    try:
        import equiload
    except ImportError as error:
        raise RuntimeError(
            '--torch-loader equiload requires the separately installed EquiLoad extension. '
            "Build it against this environment's installed Torch with "
            "EquiLoad's scripts/install-python.sh --venv /path/to/.venv, then rerun "
            'Equitrain with --torch-loader equiload.'
        ) from error
    return equiload


class EquiLoadGraphLoader:
    """Adapt EquiLoad's native loader to Equitrain's list-of-PyG-batches contract."""

    _uses_equiload = True

    def __init__(self, native_loader: Any, device: torch.device) -> None:
        self._native_loader = native_loader
        self.dataset = native_loader.dataset
        self._device = device

    def __len__(self) -> int:
        return len(self._native_loader)

    def __iter__(self) -> Iterator[list[Any]]:
        non_blocking = self._device.type == 'cuda'
        for batches in self._native_loader:
            yield [
                batch.to(self._device, non_blocking=non_blocking) for batch in batches
            ]


def get_equiload_dataloader(
    *,
    files: list[Path],
    atomic_numbers: list[int],
    r_max: float,
    args: Any,
    accelerator: Any = None,
) -> EquiLoadGraphLoader:
    """Create a single-process native HDF5 graph loader.

    EquiLoad performs loading/collation natively and does not implement PyTorch
    workers, distributed sharding, reaction grouping, or weighted resampling.
    """
    process_count = int(getattr(accelerator, 'num_processes', 1) or 1)
    if process_count != 1:
        raise RuntimeError(
            '--torch-loader equiload currently supports single-process training only; '
            'use --torch-loader python for distributed training.'
        )
    equiload = _import_equiload()
    device = getattr(accelerator, 'device', torch.device('cpu'))
    native_loader = equiload.HDF5GraphLoader.from_hdf5_files(
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
    return EquiLoadGraphLoader(native_loader, torch.device(device))
