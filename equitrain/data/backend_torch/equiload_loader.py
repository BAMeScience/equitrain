"""Optional adapter for the separately installed EquiLoad HDF5 graph loader."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np
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
            'Check out tag v0.1.0 from https://github.com/BAMeScience/EquiLoad '
            "and build it against this environment's installed Torch with "
            'scripts/install-python.sh --venv /path/to/.venv, then rerun '
            'Equitrain with --torch-loader equiload.'
        ) from error
    return equiload


class EquiLoadGraphLoader:
    """Adapt EquiLoad's native loader to Equitrain's list-of-PyG-batches contract."""

    _uses_equiload = True

    def __init__(
        self, native_loader: Any, device: torch.device, files: list[Path] | None = None
    ) -> None:
        self._native_loader = native_loader
        self.dataset = native_loader.dataset
        self._device = device
        self._file_ranges = []
        offset = 0
        for path in files or []:
            with h5py.File(path, 'r') as handle:
                end = offset + len(handle['structures'])
            self._file_ranges.append((path, offset, end))
            offset = end

    def __len__(self) -> int:
        return len(self._native_loader)

    def __iter__(self) -> Iterator[list[Any]]:
        non_blocking = self._device.type == 'cuda'
        for batches in self._native_loader:
            for batch in batches:
                self._attach_target_weights(batch)
            yield [
                batch.to(self._device, non_blocking=non_blocking) for batch in batches
            ]

    def _attach_target_weights(self, batch):
        # Older native loaders omit these fields. Read only the selected metadata
        # rows instead of materializing weights for an entire large dataset.
        missing = [
            key
            for key in ('energy_weight', 'forces_weight', 'stress_weight')
            if getattr(batch, key, None) is None
        ]
        if not missing or not self._file_ranges:
            return
        indices = batch.idx.detach().cpu().numpy().reshape(-1)
        weights = {key: np.ones(len(indices)) for key in missing}
        for path, start, end in self._file_ranges:
            positions = np.flatnonzero((indices >= start) & (indices < end))
            if not len(positions):
                continue
            local_indices, inverse = np.unique(
                indices[positions] - start, return_inverse=True
            )
            with h5py.File(path, 'r') as handle:
                structures = handle['structures']
                available = [key for key in missing if key in structures.dtype.names]
                if available:
                    rows = structures.fields(available)[local_indices]
                    for key in available:
                        weights[key][positions] = rows[key][inverse]
        for key, values in weights.items():
            setattr(
                batch,
                key,
                torch.as_tensor(values, dtype=batch.y.dtype, device=batch.y.device),
            )


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
    return EquiLoadGraphLoader(native_loader, torch.device(device), files)
