
import ast
import logging
import numpy as np
import torch

from e3nn.util.jit import compile_mode
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .atomic        import AtomicNumberTable
from .configuration import Configurations
from .utility       import compute_one_hot, to_numpy
from .scatter       import scatter_sum


@compile_mode("script")
class AtomicEnergiesBlock(torch.nn.Module):
    atomic_energies: torch.Tensor

    def __init__(self, atomic_energies: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        assert len(atomic_energies.shape) == 1

        self.register_buffer(
            "atomic_energies",
            torch.tensor(atomic_energies, dtype=torch.get_default_dtype()),
        )  # [n_elements, ]

    def forward(
        self, x: torch.Tensor  # one-hot of elements [..., n_elements]
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(x, self.atomic_energies)

    def __repr__(self):
        formatted_energies = ", ".join([f"{x:.4f}" for x in self.atomic_energies])
        return f"{self.__class__.__name__}(energies=[{formatted_energies}])"


def get_atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    z_set = set()
    for z in zs:
        z_set.add(z)
    return AtomicNumberTable(sorted(list(z_set)))


def get_atomic_energies(E0s, train_collection, z_table) -> dict:
    if E0s is not None:
        logging.info(
            "Atomic Energies not in training file, using command line argument E0s"
        )
        if E0s.lower() == "average":
            logging.info(
                "Computing average Atomic Energies using least squares regression"
            )
            # catch if colections.train not defined above
            try:
                assert train_collection is not None
                atomic_energies_dict = compute_average_E0s(
                    train_collection, z_table
                )
            except Exception as e:
                raise RuntimeError(
                    f"Could not compute average E0s if no training xyz given, error {e} occured"
                ) from e
        else:
            try:
                atomic_energies_dict = ast.literal_eval(E0s)
                assert isinstance(atomic_energies_dict, dict)
            except Exception as e:
                raise RuntimeError(
                    f"E0s specified invalidly, error {e} occured"
                ) from e
    else:
        raise RuntimeError(
            "E0s not found in training file and not specified in command line"
        )
    return atomic_energies_dict


def compute_statistics(
    data_loader    : torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
    z_table        : AtomicNumberTable
) -> Tuple[float, float, float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    atom_energy_list = []
    forces_list = []
    num_neighbors = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(compute_one_hot(z_table, batch))
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        )
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_energy_list.append(
            (batch.y - graph_e0s) / graph_sizes
        )  # {[n_graphs], }
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


def compute_average_E0s(
    collections_train: Configurations, z_table: AtomicNumberTable
) -> Dict[int, float]:
    """
    Function to compute the average interaction energy of each chemical element
    returns dictionary of E0s
    """
    len_train = len(collections_train)
    len_zs = len(z_table)
    A = np.zeros((len_train, len_zs))
    B = np.zeros(len_train)
    for i in range(len_train):
        B[i] = collections_train[i].energy
        for j, z in enumerate(z_table.zs):
            A[i, j] = np.count_nonzero(collections_train[i].atomic_numbers == z)
    try:
        E0s = np.linalg.lstsq(A, B, rcond=None)[0]
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = E0s[i]
    except np.linalg.LinAlgError:
        logging.warning(
            "Failed to compute E0s using least squares regression, using the same for all atoms"
        )
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = 0.0
    return atomic_energies_dict
