###########################################################################################
# Data parsing utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import ase.data
import ase.io
import numpy as np

import logging
import h5py
import torch

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .atomic import AtomicNumberTable


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()


def random_train_valid_split(
    items: Sequence, valid_fraction: float, seed: int
) -> Tuple[List, List]:
    assert 0.0 < valid_fraction < 1.0

    size = len(items)
    train_size = size - int(valid_fraction * size)

    indices = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    return (
        [items[i] for i in indices[:train_size]],
        [items[i] for i in indices[train_size:]],
    )


def atomic_numbers_to_indices(
    atomic_numbers: np.ndarray,
    z_table       : AtomicNumberTable
) -> np.ndarray:
    to_index_fn = np.vectorize(z_table.z_to_index)
    return to_index_fn(atomic_numbers)


def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Generates one-hot encoding with <num_classes> classes from <indices>
    :param indices: (N x 1) tensor
    :param num_classes: number of classes
    :param device: torch device
    :return: (N x num_classes) tensor
    """
    shape = indices.shape[:-1] + (num_classes,)
    oh = torch.zeros(shape, device=indices.device, dtype=torch.get_default_dtype()).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)


def compute_one_hot(z_table, batch):
    indices = atomic_numbers_to_indices(batch.atomic_numbers, z_table=z_table)
    one_hot = to_one_hot(
        torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
        num_classes=len(z_table),
    )
    return one_hot
