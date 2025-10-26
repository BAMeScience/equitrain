from __future__ import annotations

import numpy as np
import torch

from ..atomic import AtomicNumberTable


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()


def atomic_numbers_to_indices(
    atomic_numbers_tensor: torch.Tensor, atomic_numbers: AtomicNumberTable
) -> np.ndarray:
    to_index_fn = np.vectorize(atomic_numbers.z_to_index)
    return to_index_fn(atomic_numbers_tensor)


def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Generates one-hot encoding with <num_classes> classes from <indices>
    :param indices: (N x 1) tensor
    :param num_classes: number of classes
    :return: (N x num_classes) tensor
    """
    shape = indices.shape[:-1] + (num_classes,)
    oh = torch.zeros(
        shape, device=indices.device, dtype=torch.get_default_dtype()
    ).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)


def compute_one_hot(batch, atomic_numbers):
    indices = atomic_numbers_to_indices(batch, atomic_numbers)
    one_hot = to_one_hot(
        torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
        num_classes=len(atomic_numbers),
    )
    return one_hot


__all__ = [
    'to_numpy',
    'atomic_numbers_to_indices',
    'to_one_hot',
    'compute_one_hot',
]
