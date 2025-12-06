from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from equitrain.data.format_hdf5 import HDF5Dataset

from .loaders_impl import GraphDataLoader


def get_dataloader(
    *,
    data_file: Path | str | Sequence[Path | str],
    atomic_numbers,
    r_max: float,
    shuffle: bool,
    max_nodes: int | None,
    max_edges: int | None,
    drop: bool | None = None,  # kept for API compatibility
    seed: int | None = None,
    niggli_reduce: bool = False,
    max_batches: int | None = None,
    prefetch_batches: int | None = None,
    num_workers: int | None = None,
    graph_multiple: int | None = None,
):
    del drop  # UNUSED: legacy option from torch backend
    if data_file is None:
        return None

    if isinstance(data_file, list | tuple):
        files = list(data_file)
    else:
        files = [data_file]
    datasets = [HDF5Dataset(Path(file), mode='r') for file in files]

    return GraphDataLoader(
        datasets=datasets,
        z_table=atomic_numbers,
        r_max=r_max,
        n_node=max_nodes,
        n_edge=max_edges,
        shuffle=shuffle,
        seed=seed,
        niggli_reduce=niggli_reduce,
        max_batches=max_batches,
        prefetch_batches=prefetch_batches,
        num_workers=num_workers,
        graph_multiple=graph_multiple,
    )


def get_dataloaders(args, atomic_numbers, r_max):
    prefetch = getattr(args, 'prefetch_batches', None)
    num_workers = getattr(args, 'num_workers', 0)
    train_loader = get_dataloader(
        data_file=args.train_file,
        atomic_numbers=atomic_numbers,
        r_max=r_max,
        shuffle=args.shuffle,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
        prefetch_batches=prefetch,
        num_workers=num_workers,
        graph_multiple=None,
    )
    valid_loader = get_dataloader(
        data_file=args.valid_file,
        atomic_numbers=atomic_numbers,
        r_max=r_max,
        shuffle=False,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
        prefetch_batches=prefetch,
        num_workers=num_workers,
        graph_multiple=None,
    )
    test_loader = get_dataloader(
        data_file=args.test_file,
        atomic_numbers=atomic_numbers,
        r_max=r_max,
        shuffle=False,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
        prefetch_batches=prefetch,
        num_workers=num_workers,
        graph_multiple=None,
    )
    return train_loader, valid_loader, test_loader
