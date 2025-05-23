from abc import ABC, abstractmethod

import dgl
import torch

from equitrain.data.atomic import AtomicNumberTable
from equitrain.derivatives.force import compute_force
from equitrain.derivatives.stress import compute_stress, get_displacement


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
            from mace.modules.blocks import RadialEmbeddingBlock
            from mace.modules.radial import (
                AgnesiTransform,
                BesselBasis,
                ChebychevBasis,
                GaussianBasis,
                SoftTransform,
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
            from mace.modules.radial import ZBLBasis

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


class M3GNetWrapper(AbstractWrapper):
    """
    Wrapper for M3GNet models to be used with Equitrain.

    This wrapper integrates the Materials 3-body Graph Network (M3GNet) potential from the MatGL
    library into the Equitrain framework. It supports energy, forces, and stress prediction workflows.

    Parameters
    ----------
    args : object
        Arguments object containing training parameters
    model : torch.nn.Module, optional
        A pre-trained M3GNet model. If None, a new model will be created.
    element_types : list[str], optional
        List of chemical symbols in order. Default is None, which will use the model's
        element_types if available.
    """

    def __init__(self, args, model=None, element_types=None):
        """
        Initialize the M3GNet wrapper.

        Parameters
        ----------
        args : object
            Arguments object containing training parameters
        model : torch.nn.Module, optional
            A pre-trained M3GNet model. If None, a new model will be created.
        element_types : list[str], optional
            List of chemical symbols in order. Default is None, which will use the model's
            element_types if available.
        """
        super().__init__(model)

        # Set compute flags based on loss weights
        self.compute_force = args.forces_weight > 0.0
        self.compute_stress = args.stress_weight > 0.0

        # If element_types is provided, use it; otherwise, use the model's element_types
        if element_types is not None:
            self._element_types = element_types
        else:
            self._element_types = self.model.element_types

        # Create a converter for structures to graphs
        try:
            from matgl.ext.pymatgen import Structure2Graph

            self._graph_converter = Structure2Graph(
                element_types=self._element_types, cutoff=self.r_max
            )
        except ImportError:
            self._graph_converter = None

    def forward(self, *args):
        """
        Forward pass through the M3GNet model.

        Parameters
        ----------
        *args : tuple
            Input data. Can be a PyTorch Geometric Data object or a tuple of
            (species, coordinates).

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing 'energy', 'forces', and 'stress' predictions.
        """
        # Handle different input formats
        if len(args) == 1:
            # PyG Data object
            data = args[0]

            # Create a DGL graph from the PyG data
            num_graphs = data.ptr.shape[0] - 1

            # Extract positions and make them require gradients if needed
            positions = data.positions.requires_grad_(
                self.compute_force or self.compute_stress
            )

            # Create a DGL graph
            g = dgl.graph((data.edge_index[0], data.edge_index[1]))
            g.ndata['atomic_numbers'] = data.atomic_numbers
            g.ndata['pos'] = positions

            # Add cell information if available
            if hasattr(data, 'cell') and data.cell is not None:
                g.ndata['cell'] = data.cell

            # Add batch information
            g.ndata['batch'] = data.batch

            # If stress computation is needed, get displacement
            if self.compute_stress:
                positions, displacement = get_displacement(
                    positions, num_graphs, data.batch
                )
                g.ndata['pos'] = positions
        else:
            # Not implemented for direct tuple input yet
            raise NotImplementedError(
                'Direct tuple input is not yet implemented for M3GNetWrapper'
            )

        # Forward pass through the M3GNet model
        energy = self.model(g)

        # Prepare output dictionary
        y_pred = {'energy': energy, 'forces': None, 'stress': None}

        # Compute forces if needed
        if self.compute_force:
            forces = compute_force(energy, positions, training=self.training)
            y_pred['forces'] = forces

        # Compute stress if needed
        if self.compute_stress:
            stress = compute_stress(
                energy, displacement, data.cell, training=self.training
            )
            y_pred['stress'] = stress

        return y_pred

    @property
    def atomic_numbers(self):
        """
        Property that returns atomic numbers from the model.

        Returns
        -------
        list
            List of atomic numbers supported by the model.
        """
        return [AtomicNumberTable.from_symbol(symbol) for symbol in self._element_types]

    @property
    def atomic_energies(self):
        """
        Property that returns atomic energies from the model.

        Returns
        -------
        torch.Tensor
            Tensor of atomic energies.
        """
        # M3GNet doesn't have explicit atomic energies like ANI
        # Return zeros for compatibility
        return torch.zeros(
            len(self.atomic_numbers), device=next(self.model.parameters()).device
        )

    @property
    def r_max(self):
        """
        Property that returns the r_max value from the model.

        Returns
        -------
        float
            The r_max value.
        """
        return self.model.cutoff

    @r_max.setter
    def r_max(self, value):
        """
        Setter for r_max. Modifies the model's r_max.

        Parameters
        ----------
        value : float
            The new r_max value.
        """
        # Update the model's cutoff
        self.model.cutoff = value

        # Update the graph converter's cutoff if it exists
        if self._graph_converter is not None:
            self._graph_converter.cutoff = value
