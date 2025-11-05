"""
Lightweight fallback M3GNet wrapper that avoids optional MatGL/DGL dependencies.
"""

from __future__ import annotations

import torch

from equitrain.data.atomic import AtomicNumberTable

from .base import AbstractWrapper


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


__all__ = ['M3GNetWrapper']
