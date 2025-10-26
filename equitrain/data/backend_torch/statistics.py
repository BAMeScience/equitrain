from __future__ import annotations

import numpy as np
import torch
from e3nn.util.jit import compile_mode

from ..atomic import AtomicNumberTable
from ..format_hdf5 import HDF5Dataset
from ..statistics_data import (
    compute_average_atomic_energies as _compute_average_atomic_energies,
)
from .scatter import scatter_sum
from .utility import compute_one_hot, to_numpy


@compile_mode('script')
class AtomicEnergiesBlock(torch.nn.Module):
    atomic_energies: torch.Tensor

    def __init__(self, atomic_energies: np.ndarray | torch.Tensor):
        super().__init__()
        assert len(atomic_energies.shape) == 1

        self.register_buffer(
            'atomic_energies',
            torch.tensor(atomic_energies, dtype=torch.get_default_dtype()),
        )  # [n_elements, ]

    def forward(
        self,
        x: torch.Tensor,  # one-hot of elements [..., n_elements]
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(x, self.atomic_energies)

    def __repr__(self):
        formatted_energies = ', '.join([f'{x:.4f}' for x in self.atomic_energies])
        return f'{self.__class__.__name__}(energies=[{formatted_energies}])'


def compute_statistics(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: dict,
    atomic_numbers: AtomicNumberTable,
) -> tuple[float, float, float]:
    atomic_energies_list: np.ndarray = np.array([
        atomic_energies[z] for z in atomic_numbers
    ])

    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies_list)

    atom_energy_list = []
    forces_list = []
    num_neighbors = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(
            compute_one_hot(batch.atomic_numbers, atomic_numbers)
        )
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        )
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_energy_list.append((batch.y - graph_e0s) / graph_sizes)  # {[n_graphs], }
        forces_list.append(batch.force)  # {[n_graphs*n_atoms,3], }

        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }

    mean = to_numpy(torch.mean(atom_energies)).item()
    rms = to_numpy(torch.sqrt(torch.mean(torch.square(forces)))).item()

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )

    return to_numpy(avg_num_neighbors).item(), mean, rms


def compute_atomic_numbers(
    dataset: HDF5Dataset,
) -> AtomicNumberTable:
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        # shuffle data in case we only use a subset of the data
        shuffle=True,
        drop_last=False,
        pin_memory=False,
        num_workers=2,
        collate_fn=lambda data: data,
    )

    z_set = set()

    for batch in data_loader:
        # Convert from int64 to int32, which is json serializable
        z_set.update([int(z) for z in batch[0].get_atomic_numbers()])

    return AtomicNumberTable(sorted(list(z_set)))


def compute_average_atomic_energies(
    dataset: HDF5Dataset,
    z_table: AtomicNumberTable,
    max_n: int = None,
) -> dict[int, float]:
    return _compute_average_atomic_energies(dataset, z_table, max_n=max_n)


__all__ = [
    'AtomicNumberTable',
    'AtomicEnergiesBlock',
    'compute_statistics',
    'compute_atomic_numbers',
    'compute_average_atomic_energies',
]
