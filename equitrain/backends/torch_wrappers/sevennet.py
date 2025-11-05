"""
SevenNet wrapper.
"""

from __future__ import annotations

import torch

from equitrain.data.atomic import AtomicNumberTable

from .base import AbstractWrapper


class SevennetWrapper(AbstractWrapper):
    def __init__(self, args, model):
        super().__init__(model)

    def forward(self, input):
        input.energy = input.y
        input.forces = input['force']
        input.edge_vec, _ = self.get_edge_vectors_and_lengths(
            input.positions, input.edge_index, input.shifts
        )
        input.num_atoms = input.ptr[1:] - input.ptr[:-1]

        y_pred = self.model(input)

        y_pred = {
            'energy': y_pred.inferred_total_energy,
            'forces': y_pred.inferred_force,
            'stress': self.batch_voigt_to_tensor(y_pred.inferred_stress).type(
                y_pred.inferred_total_energy.dtype
            ),
        }

        return y_pred

    @classmethod
    def get_edge_vectors_and_lengths(
        cls,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        vectors = positions[receiver] - positions[sender] + shifts
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        return vectors, lengths

    @classmethod
    def batch_voigt_to_tensor(cls, voigts: torch.Tensor) -> torch.Tensor:
        tensors = torch.zeros(
            (voigts.shape[0], 3, 3), dtype=voigts.dtype, device=voigts.device
        )
        tensors[:, 0, 0] = voigts[:, 0]
        tensors[:, 1, 1] = voigts[:, 1]
        tensors[:, 2, 2] = voigts[:, 2]
        tensors[:, 1, 2] = tensors[:, 2, 1] = voigts[:, 3]
        tensors[:, 0, 2] = tensors[:, 2, 0] = voigts[:, 4]
        tensors[:, 0, 1] = tensors[:, 1, 0] = voigts[:, 5]
        return tensors

    @property
    def atomic_numbers(self):
        return AtomicNumberTable(
            torch.nonzero(self.model.z_to_onehot_tensor != -1).squeeze().cpu().tolist()
        )

    @property
    def atomic_energies(self):
        return None

    @property
    def r_max(self):
        return self.model.cutoff.item()

    @r_max.setter
    def r_max(self, value):
        self.model.cutoff.fill_(value)


__all__ = ['SevennetWrapper']
