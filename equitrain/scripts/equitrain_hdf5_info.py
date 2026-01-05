#!/usr/bin/env python3
"""Inspect Equitrain HDF5 datasets and summarize layout/size."""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import numpy as np

from equitrain.data.format_hdf5 import HDF5Dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Inspect Equitrain HDF5 datasets.',
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help='HDF5 files, directories, or glob patterns.',
    )
    parser.add_argument(
        '--max-entries',
        type=int,
        default=None,
        help='Limit per-file atom count statistics to the first N entries.',
    )
    return parser.parse_args()


def _format_bytes(size: int) -> str:
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f'{value:.1f} {unit}'
        value /= 1024.0
    return f'{value:.1f} {units[-1]}'


def _expand_paths(raw_paths: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for raw in raw_paths:
        if any(ch in raw for ch in '*?['):
            expanded.extend(Path(path) for path in glob.glob(raw))
            continue
        path = Path(raw)
        if path.is_dir():
            expanded.extend(sorted(path.glob('*.h5')))
            expanded.extend(sorted(path.glob('*.hdf5')))
            continue
        expanded.append(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in expanded:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _describe_dataset(handle, name: str) -> None:
    if name not in handle:
        return
    dataset = handle[name]
    chunks = dataset.chunks if dataset.chunks is not None else 'contiguous'
    compression = dataset.compression or 'none'
    logging.info(
        '  %s: shape=%s dtype=%s chunks=%s compression=%s',
        name,
        dataset.shape,
        dataset.dtype,
        chunks,
        compression,
    )


def _log_hdf5_info(path: Path, *, max_entries: int | None) -> None:
    logging.info('HDF5 file: %s', path)
    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0
    if size_bytes:
        logging.info('  size: %s', _format_bytes(size_bytes))

    with HDF5Dataset(path, mode='r') as dataset:
        total_entries = len(dataset)
        logging.info('  entries: %s', total_entries)

        handle = dataset.file
        logging.info('  datasets: %s', ', '.join(sorted(handle.keys())))

        _describe_dataset(handle, dataset.STRUCTURES_DATASET)
        _describe_dataset(handle, dataset.POSITIONS_DATASET)
        _describe_dataset(handle, dataset.FORCES_DATASET)
        _describe_dataset(handle, dataset.ATOMIC_NUMBERS_DATASET)

        structures = handle[dataset.STRUCTURES_DATASET]
        if structures.dtype.names:
            logging.info('  structures fields: %s', ', '.join(structures.dtype.names))

        if total_entries == 0:
            return

        lengths = structures['length']
        sample_entries = total_entries
        if max_entries is not None:
            sample_entries = min(int(max_entries), total_entries)
            lengths = lengths[:sample_entries]
        lengths = np.asarray(lengths, dtype=np.int64)
        if lengths.size:
            logging.info(
                '  atoms_per_entry: min=%s mean=%.1f max=%s%s',
                int(lengths.min()),
                float(lengths.mean()),
                int(lengths.max()),
                ''
                if sample_entries == total_entries
                else f' (sample={sample_entries})',
            )

        if dataset.POSITIONS_DATASET in handle:
            total_atoms = int(handle[dataset.POSITIONS_DATASET].shape[0])
            logging.info('  atoms_total: %s', total_atoms)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    expanded = _expand_paths(args.paths)
    if not expanded:
        raise ValueError('No HDF5 files found.')

    for idx, path in enumerate(expanded):
        if idx:
            logging.info('')
        _log_hdf5_info(path, max_entries=args.max_entries)


if __name__ == '__main__':
    main()
