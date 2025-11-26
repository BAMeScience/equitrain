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
    batch_size: int | None,
    shuffle: bool,
    max_nodes: int | None,
    max_edges: int | None,
    drop: bool | None = None,  # kept for API compatibility
    seed: int | None = None,
    niggli_reduce: bool = False,
    max_batches: int | None = None,
):
    del drop  # UNUSED: legacy option from torch backend
    if data_file is None:
        return None

    if isinstance(data_file, (list, tuple)):
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
        n_graph=batch_size,
        shuffle=shuffle,
        seed=seed,
        niggli_reduce=niggli_reduce,
        max_batches=max_batches,
    )


def get_dataloaders(args, atomic_numbers, r_max):
    train_loader = get_dataloader(
        data_file=args.train_file,
        atomic_numbers=atomic_numbers,
        r_max=r_max,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
    )
    valid_loader = get_dataloader(
        data_file=args.valid_file,
        atomic_numbers=atomic_numbers,
        r_max=r_max,
        batch_size=args.batch_size,
        shuffle=False,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
    )
    test_loader = get_dataloader(
        data_file=args.test_file,
        atomic_numbers=atomic_numbers,
        r_max=r_max,
        batch_size=args.batch_size,
        shuffle=False,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
    )
    return train_loader, valid_loader, test_loader
