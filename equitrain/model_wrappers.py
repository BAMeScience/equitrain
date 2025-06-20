from abc import ABC, abstractmethod

import torch

from equitrain.data.atomic import AtomicNumberTable

try:
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
except ImportError:
    _HAS_MACE = False


class AbstractWrapper(torch.nn.Module, ABC):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @abstractmethod
    def forward(self, *args):
        """
        Defines the forward pass. Should implement the forward pass for the model.
        """
        pass

    @property
    @abstractmethod
    def atomic_numbers(self):
        """
        Property that should return atomic numbers from the model.
        """
        pass

    @property
    @abstractmethod
    def atomic_energies(self):
        """
        Property that should return atomic numbers from the model.
        """
        pass

    @property
    @abstractmethod
    def r_max(self):
        """
        Property that should return the r_max value from the model.
        """
        pass

    @r_max.setter
    @abstractmethod
    def r_max(self, value):
        """
        Setter for r_max. Should modify the model's r_max.
        """
        pass


class MaceWrapper(AbstractWrapper):
    def __init__(self, args, model, optimize_atomic_energies=False):
        super().__init__(model)

        if optimize_atomic_energies:
            if 'atomic_energies' in self.model.atomic_energies_fn._buffers:
                atomic_energies = self.model.atomic_energies_fn.atomic_energies
                del self.model.atomic_energies_fn._buffers['atomic_energies']
                self.model.atomic_energies_fn.atomic_energies = torch.nn.Parameter(
                    atomic_energies
                )

        self.compute_force = args.forces_weight > 0.0
        self.compute_stress = args.stress_weight > 0.0

    def forward(self, *args):
        y_pred = self.model(
            *args,
            compute_force=self.compute_force,
            compute_stress=self.compute_stress,
            training=self.training,
        )

        if not isinstance(y_pred, dict):
            y_pred = {'energy': y_pred[0], 'forces': y_pred[1], 'stress': y_pred[2]}

        return y_pred

    @property
    def atomic_numbers(self):
        return AtomicNumberTable(self.model.atomic_numbers.tolist())

    @property
    def atomic_energies(self):
        return self.model.atomic_energies_fn.atomic_energies.tolist()

    @property
    def r_max(self):
        return self.model.r_max.item()

    @r_max.setter
    def r_max(self, r_max):
        if hasattr(self.model, 'radial_embedding'):
            if _HAS_MACE is False:
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
            if _HAS_MACE is False:
                raise ImportError(
                    "Optional dependency 'mace' is required for MaceWrapper."
                )

            if self.model.pair_repulsion:
                p = self.model.pair_repulsion_fn.p
                self.model.pair_repulsion_fn = ZBLBasis(r_max=r_max, p=p)

        self.model.r_max.fill_(r_max)


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
        positions: torch.Tensor,  # [n_nodes, 3]
        edge_index: torch.Tensor,  # [2, n_edges]
        shifts: torch.Tensor,  # [n_edges, 3]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]

        return vectors, lengths

    @classmethod
    def batch_voigt_to_tensor(cls, voigts):
        """
        Convert a batch of Voigt notation arrays back to 3x3 stress tensors.

        Parameters:
            voigts (torch.Tensor): Tensor of shape (N, 6) representing N Voigt stress vectors.
                                Gradients will be preserved if attached.

        Returns:
            torch.Tensor: Tensor of shape (N, 3, 3) with full stress tensors.
        """
        tensors = torch.zeros(
            (voigts.shape[0], 3, 3), dtype=voigts.dtype, device=voigts.device
        )
        tensors[:, 0, 0] = voigts[:, 0]  # σ_xx
        tensors[:, 1, 1] = voigts[:, 1]  # σ_yy
        tensors[:, 2, 2] = voigts[:, 2]  # σ_zz
        tensors[:, 1, 2] = tensors[:, 2, 1] = voigts[:, 3]  # σ_yz
        tensors[:, 0, 2] = tensors[:, 2, 0] = voigts[:, 4]  # σ_xz
        tensors[:, 0, 1] = tensors[:, 1, 0] = voigts[:, 5]  # σ_xy
        return tensors

    @property
    def atomic_numbers(self):
        return AtomicNumberTable(
            torch.nonzero(self.model.z_to_onehot_tensor != -1).squeeze().tolist()
        )

    @property
    def atomic_energies(self):
        return None

    @property
    def r_max(self):
        return self.model.cutoff

    @r_max.setter
    def r_max(self, value):
        self.model.cutoff.fill_(value)
