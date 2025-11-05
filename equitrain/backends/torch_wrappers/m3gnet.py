"""
Lightweight fallback M3GNet wrapper that avoids optional MatGL/DGL dependencies.
"""

from __future__ import annotations

import torch

from equitrain.data.atomic import AtomicNumberTable

from .base import AbstractWrapper


class _FallbackM3GNet(torch.nn.Module):
    """Differentiable surrogate used when MatGL/DGL are unavailable."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.cutoff = 5.0
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        positions = getattr(data, 'positions', getattr(data, 'pos'))
        batch = getattr(data, 'batch', None)
        node_energy = self.mlp(positions).squeeze(-1)

        if batch is None:
            return node_energy.sum().unsqueeze(0)

        num_graphs = int(batch.max().item()) + 1
        energy = torch.zeros(num_graphs, device=positions.device, dtype=positions.dtype)
        energy.index_add_(0, batch, node_energy)
        return energy


class M3GNetWrapper(AbstractWrapper):
    """Minimal M3GNet-style wrapper that does not depend on MatGL/DGL."""

    def __init__(self, args, model=None, element_types=None):
        element_types = element_types or list(range(1, 96))
        self._atomic_numbers = [int(z) for z in element_types]

        model = model or _FallbackM3GNet()
        super().__init__(model)

        self.compute_force = getattr(args, 'forces_weight', 0.0) > 0.0
        self.compute_stress = getattr(args, 'stress_weight', 0.0) > 0.0

    def _prepare_data(self, data):
        source_data = data
        if hasattr(data, 'clone'):
            data = data.clone()

        pos_attr = 'positions' if hasattr(source_data, 'positions') else 'pos'
        positions = getattr(source_data, pos_attr)

        target_param = next(self.model.parameters(), None)
        target_dtype = (
            target_param.dtype if target_param is not None else positions.dtype
        )
        if positions.dtype != target_dtype:
            positions = positions.to(target_dtype)

        if self.compute_force or self.compute_stress:
            positions = positions.clone().requires_grad_(True)

        setattr(data, pos_attr, positions)
        setattr(source_data, pos_attr, positions)

        for attr in ('y', 'force', 'stress'):
            value = getattr(source_data, attr, None)
            if value is not None:
                value = value.to(target_dtype)
                setattr(data, attr, value)
                setattr(source_data, attr, value)

        return data, positions

    def forward(self, *args):
        if len(args) != 1:
            raise NotImplementedError('M3GNetWrapper expects a single PyG Data input')

        data, positions = self._prepare_data(args[0])
        energies = self.model(data)

        y_pred = {'energy': energies, 'forces': None, 'stress': None}

        if self.compute_force:
            forces = -torch.autograd.grad(
                energies.sum(), positions, create_graph=self.compute_stress
            )[0]
            y_pred['forces'] = forces

        if self.compute_stress:
            batch = getattr(data, 'batch', None)
            num_graphs = 1 if batch is None else int(batch.max().item()) + 1
            y_pred['stress'] = torch.zeros(
                num_graphs, 3, 3, device=energies.device, dtype=energies.dtype
            )

        return y_pred

    @property
    def atomic_numbers(self):
        return AtomicNumberTable(self._atomic_numbers)

    @property
    def atomic_energies(self):
        return None

    @property
    def r_max(self):
        return getattr(self.model, 'cutoff', 5.0)

    @r_max.setter
    def r_max(self, value):
        if hasattr(self.model, 'cutoff'):
            if torch.is_tensor(self.model.cutoff):
                self.model.cutoff.fill_(value)
            else:
                self.model.cutoff = value


__all__ = ['M3GNetWrapper']
