"""
MACE-specific torch wrapper.
"""

from __future__ import annotations

import torch

from equitrain.data.atomic import AtomicNumberTable

from .base import AbstractWrapper

try:  # pragma: no cover - optional dependency
    from mace.modules.blocks import RadialEmbeddingBlock
    from mace.modules.radial import (
        AgnesiTransform,
        BesselBasis,
        ChebychevBasis,
        GaussianBasis,
        SoftTransform,
        ZBLBasis,
    )

    _HAS_MACE = True
except Exception:  # pragma: no cover - guard optional import errors
    _HAS_MACE = False


class MaceWrapper(AbstractWrapper):
    def __init__(self, args, model, optimize_atomic_energies: bool = False):
        super().__init__(model)

        if optimize_atomic_energies:
            if 'atomic_energies' in self.model.atomic_energies_fn._buffers:
                atomic_energies = self.model.atomic_energies_fn.atomic_energies
                del self.model.atomic_energies_fn._buffers['atomic_energies']
                self.model.atomic_energies_fn.atomic_energies = torch.nn.Parameter(
                    atomic_energies
                )

        self.compute_force = getattr(args, 'forces_weight', 0.0) > 0.0
        self.compute_stress = getattr(args, 'stress_weight', 0.0) > 0.0

    def forward(self, *args):
        if len(args) != 1:
            raise NotImplementedError('MaceWrapper expects a single PyG batch argument.')

        data = args[0]
        param = next(self.model.parameters(), None)
        target_device = param.device if param is not None else None
        target_dtype = param.dtype if param is not None else None

        if target_device is not None:
            data = data.to(target_device)
        if target_dtype is not None:
            for key, value in data:
                if isinstance(value, torch.Tensor) and value.dtype.is_floating_point and value.dtype != target_dtype:
                    setattr(data, key, value.to(dtype=target_dtype))

        y_pred = self.model(
            data,
            compute_force=self.compute_force,
            compute_stress=self.compute_stress,
            training=self.training,
        )

        if not isinstance(y_pred, dict):
            y_pred = {'energy': y_pred[0], 'forces': y_pred[1], 'stress': y_pred[2]}

        return y_pred

    @property
    def atomic_numbers(self):
        return AtomicNumberTable(self.model.atomic_numbers.cpu().tolist())

    @property
    def atomic_energies(self):
        return self.model.atomic_energies_fn.atomic_energies.cpu().tolist()

    @property
    def r_max(self):
        return self.model.r_max.item()

    @r_max.setter
    def r_max(self, r_max):
        if hasattr(self.model, 'radial_embedding'):
            if not _HAS_MACE:
                raise ImportError(
                    "Optional dependency 'mace' is required for MaceWrapper."
                )

            num_bessel = self.model.radial_embedding.out_dim
            num_polynomial_cutoff = self.model.radial_embedding.cutoff_fn.p.item()

            if isinstance(self.model.radial_embedding.bessel_fn, BesselBasis):
                radial_type = 'bessel'
            elif isinstance(self.model.radial_embedding.bessel_fn, ChebychevBasis):
                radial_type = 'chebychev'
            elif isinstance(self.model.radial_embedding.bessel_fn, GaussianBasis):
                radial_type = 'gaussian'
            else:
                return

            if isinstance(
                self.model.radial_embedding.distance_transform, AgnesiTransform
            ):
                distance_transform = 'Agnesi'
            elif isinstance(
                self.model.radial_embedding.distance_transform, SoftTransform
            ):
                distance_transform = 'Soft'
            else:
                return

            self.model.radial_embedding = RadialEmbeddingBlock(
                r_max=r_max,
                num_bessel=num_bessel,
                num_polynomial_cutoff=num_polynomial_cutoff,
                radial_type=radial_type,
                distance_transform=distance_transform,
            )

        if hasattr(self.model, 'pair_repulsion'):
            if not _HAS_MACE:
                raise ImportError(
                    "Optional dependency 'mace' is required for MaceWrapper."
                )

            if self.model.pair_repulsion:
                p = self.model.pair_repulsion_fn.p
                self.model.pair_repulsion_fn = ZBLBasis(r_max=r_max, p=p)

        self.model.r_max.fill_(r_max)


__all__ = ['MaceWrapper']
