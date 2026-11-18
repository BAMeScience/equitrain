"""Adapt native GRACE to Equitrain's Torch graph and derivative interface."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch

from equitrain.data.atomic import AtomicNumberTable
from equitrain.models.grace import Grace1L

from .base import AbstractWrapper


class GraceWrapper(AbstractWrapper):
    def __init__(self, args, model):
        if not isinstance(model, Grace1L):
            raise TypeError('GraceWrapper requires a native Grace1L model.')
        super().__init__(model)
        self.compute_force = getattr(args, 'forces_weight', 0.0) > 0
        self.compute_stress = getattr(args, 'stress_weight', 0.0) > 0

    def forward(self, data):
        derivatives = self.compute_force or self.compute_stress
        if derivatives and torch.is_inference_mode_enabled():
            raise RuntimeError(
                'GRACE derivatives require no_grad or normal mode, not inference_mode.'
            )
        with torch.enable_grad() if derivatives else nullcontext():
            return self._forward(data)

    def _forward(self, data):
        parameter = self.model.embedding
        device, dtype = parameter.device, parameter.dtype
        positions = data.positions.to(device=device, dtype=dtype)
        if self.compute_force:
            positions = positions.clone().requires_grad_(True)
        numbers = data.atomic_numbers.to(device=device)
        matches = numbers[:, None] == self.model.atomic_numbers[None, :]
        if not torch.all(matches.any(-1)):
            raise ValueError(
                'Graph contains an element unsupported by this GRACE model.'
            )
        species = matches.long().argmax(-1)
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(len(numbers), dtype=torch.long, device=device)
            n_graphs = 1
        else:
            batch = batch.to(device=device)
            n_graphs = data.num_graphs
        centers, neighbors = data.edge_index.to(device=device)
        if torch.any(batch[centers] != batch[neighbors]):
            raise ValueError('GRACE graph contains edges between different structures.')
        vectors = (
            positions[neighbors]
            - positions[centers]
            + data.shifts.to(device=device, dtype=dtype)
        )
        strain = None
        if self.compute_stress:
            cell = data.cell.to(device=device, dtype=dtype).reshape(n_graphs, 3, 3)
            volume = torch.linalg.det(cell).abs()
            original_volume = getattr(data, 'cell_volume', volume)
            if torch.any(volume <= 1e-12) or torch.any(
                torch.as_tensor(original_volume, device=device).abs() <= 1e-12
            ):
                raise ValueError(
                    'Stress requires a nonzero cell volume for every structure.'
                )
            strain = positions.new_zeros(n_graphs, 3, 3).requires_grad_(True)
            symmetric = (strain + strain.transpose(-1, -2)) * 0.5
            # Strain the full bond vector, including the periodic image shift.
            vectors = vectors + torch.einsum(
                'ei,eij->ej', vectors, symmetric[batch[centers]]
            )
        atomic = self.model(vectors, species, centers, neighbors)
        energy = atomic.new_zeros(n_graphs).index_add(0, batch, atomic)
        forces = stress = None
        if self.compute_force or self.compute_stress:
            # Keep empty-neighbor structures connected to the geometry graph.
            total = energy.sum() + positions.sum() * 0
            inputs = []
            if self.compute_force:
                inputs.append(positions)
            if self.compute_stress:
                total = total + strain.sum() * 0
                inputs.append(strain)
            gradients = torch.autograd.grad(
                total, inputs, create_graph=self.training, retain_graph=self.training
            )
            if self.compute_force:
                forces = -gradients[0]
                if self.training:
                    forces = forces + energy.sum() * 0
            if self.compute_stress:
                stress = gradients[-1] / volume[:, None, None]
                if self.training:
                    stress = stress + energy.sum() * 0
        return {'energy': energy, 'forces': forces, 'stress': stress}

    @property
    def atomic_numbers(self):
        return AtomicNumberTable(self.model.atomic_numbers.detach().cpu().tolist())

    @property
    def atomic_energies(self):
        return (
            (
                self.model.atomic_energies
                + self.model.config.energy_shift
                + self.model.trainable_shift
            )
            .detach()
            .cpu()
            .tolist()
        )

    @property
    def r_max(self):
        return self.model.cutoff.item()

    @r_max.setter
    def r_max(self, value):
        if not math.isclose(float(value), self.r_max, rel_tol=0, abs_tol=1e-6):
            raise ValueError(
                'Construct a new Grace1L config to change its cutoff; overriding it changes the model.'
            )


__all__ = ['GraceWrapper']
