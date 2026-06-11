"""
MACE-specific torch wrapper.
"""

from __future__ import annotations

import math

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
            raise NotImplementedError(
                'MaceWrapper expects a single PyG batch argument.'
            )

        data = args[0]
        param = next(self.model.parameters(), None)
        target_device = param.device if param is not None else None
        target_dtype = param.dtype if param is not None else None

        if target_device is not None:
            data = data.to(target_device)
        if target_dtype is not None:
            for key, value in data:
                if (
                    isinstance(value, torch.Tensor)
                    and value.dtype.is_floating_point
                    and value.dtype != target_dtype
                ):
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
        r_max = float(r_max)
        if math.isclose(self.r_max, r_max, rel_tol=0.0, abs_tol=1e-6):
            return

        if hasattr(self.model, 'radial_embedding'):
            if not _HAS_MACE:
                raise ImportError(
                    "Optional dependency 'mace' is required for MaceWrapper."
                )

            radial_embedding = self.model.radial_embedding
            num_bessel = radial_embedding.out_dim
            num_polynomial_cutoff = radial_embedding.cutoff_fn.p.item()

            if isinstance(radial_embedding.bessel_fn, BesselBasis):
                radial_type = 'bessel'
                basis_attr = 'bessel_weights'
            elif isinstance(radial_embedding.bessel_fn, ChebychevBasis):
                radial_type = 'chebyshev'
                basis_attr = None
            elif isinstance(radial_embedding.bessel_fn, GaussianBasis):
                radial_type = 'gaussian'
                basis_attr = 'gaussian_weights'
            else:
                return

            if hasattr(radial_embedding, 'distance_transform'):
                if isinstance(radial_embedding.distance_transform, AgnesiTransform):
                    distance_transform = 'Agnesi'
                elif isinstance(radial_embedding.distance_transform, SoftTransform):
                    distance_transform = 'Soft'
                else:
                    return
            else:
                distance_transform = 'None'

            trainable_basis = False
            basis_requires_grad = False
            if basis_attr is not None:
                basis = getattr(radial_embedding.bessel_fn, basis_attr)
                trainable_basis = isinstance(basis, torch.nn.Parameter)
                basis_requires_grad = bool(getattr(basis, 'requires_grad', False))

            new_radial_embedding = RadialEmbeddingBlock(
                r_max=r_max,
                num_bessel=num_bessel,
                num_polynomial_cutoff=num_polynomial_cutoff,
                radial_type=radial_type,
                distance_transform=distance_transform,
                apply_cutoff=getattr(radial_embedding, 'apply_cutoff', True),
            )
            if trainable_basis and basis_attr is not None:
                new_basis = getattr(new_radial_embedding.bessel_fn, basis_attr)
                setattr(
                    new_radial_embedding.bessel_fn,
                    basis_attr,
                    torch.nn.Parameter(
                        new_basis,
                        requires_grad=basis_requires_grad,
                    ),
                )

            reference = next(radial_embedding.parameters(), None)
            if reference is None:
                reference = next(radial_embedding.buffers(), None)
            if reference is not None:
                if reference.dtype.is_floating_point:
                    new_radial_embedding = new_radial_embedding.to(
                        device=reference.device,
                        dtype=reference.dtype,
                    )
                else:
                    new_radial_embedding = new_radial_embedding.to(
                        device=reference.device
                    )
            new_radial_embedding.train(radial_embedding.training)
            self.model.radial_embedding = new_radial_embedding

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
