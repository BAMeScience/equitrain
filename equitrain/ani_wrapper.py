"""
ANI Wrapper for Equitrain

This module provides a wrapper for the TorchANI library to be used with Equitrain.
It implements the AbstractWrapper interface and provides functionality for energy-only
training workflows.
"""

import torch
import torchani

from equitrain.data.atomic import AtomicNumberTable
from equitrain.model_wrappers import AbstractWrapper


class AniWrapper(AbstractWrapper):
    """
    Wrapper for TorchANI models to be used with Equitrain.

    This wrapper integrates the Atomic Neural Network (ANI) potential from the TorchANI
    library into the Equitrain framework. It supports energy-only training workflows
    and can optionally compute forces using autograd.

    Parameters
    ----------
    args : object
        Arguments object containing training parameters
    model : torch.nn.Module, optional
        A pre-trained TorchANI model. If None, a new model will be created.
    species_order : List[str], optional
        List of chemical symbols in order (e.g., ['H', 'C', 'N', 'O']).
        Default is None, which will use the model's species order if available.
    """

    def __init__(self, args, model=None, species_order=None):
        """Initialize the ANI wrapper."""
        super().__init__(model)

        # Store arguments for later use
        self.compute_force = getattr(args, 'forces_weight', 0.0) > 0.0
        self.compute_stress = getattr(args, 'stress_weight', 0.0) > 0.0

        # If no model is provided, we need to create one
        if model is None:
            self.model = self._create_default_model(species_order)

        # Store species order for conversion
        self._species_order = species_order
        if self._species_order is None and hasattr(self.model, 'species_order'):
            self._species_order = self.model.species_order

        # Create species converter if needed
        if hasattr(self.model, 'species_converter'):
            self.species_converter = self.model.species_converter
        elif self._species_order is not None:
            self.species_converter = torchani.utils.ChemicalSymbolsToInts(
                self._species_order
            )

    def _create_default_model(self, species_order=None):
        """
        Create a default ANI model.

        This is a fallback if no model is provided. It uses ANI-1x from the model zoo.

        Parameters
        ----------
        species_order : List[str], optional
            List of chemical symbols in order. Default is None, which will use
            the model's default species order.

        Returns
        -------
        torch.nn.Module
            A TorchANI model
        """
        # Use ANI-1x as default model
        model = torchani.models.ANI1x(periodic_table_index=False)
        model.species_order = ['H', 'C', 'N', 'O']
        return model

    def forward(self, *args):
        """
        Forward pass through the ANI model.

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
            species = data.species
            coordinates = data.positions.requires_grad_(self.compute_force)
        else:
            # Tuple of (species, coordinates)
            species = args[0]
            coordinates = args[1].requires_grad_(self.compute_force)

        # Get energy prediction from the model
        _, energies = self.model((species, coordinates))

        # Prepare output dictionary
        y_pred = {'energy': energies, 'forces': None, 'stress': None}

        # Compute forces if needed
        if self.compute_force:
            forces = -torch.autograd.grad(
                energies.sum(),
                coordinates,
                create_graph=True,
                retain_graph=self.compute_stress,
            )[0]
            y_pred['forces'] = forces

        # Compute stress if needed (not implemented in ANI)
        if self.compute_stress:
            # ANI doesn't natively support stress calculation
            # Return zeros for stress with proper shape
            batch_size = energies.shape[0]
            y_pred['stress'] = torch.zeros(batch_size, 3, 3, device=energies.device)

        return y_pred

    @property
    def atomic_numbers(self):
        """
        Get the atomic numbers supported by the model.

        Returns
        -------
        AtomicNumberTable
            Table of atomic numbers supported by the model.
        """
        if hasattr(self.model, 'species_order'):
            # Convert chemical symbols to atomic numbers
            symbol_to_Z = {
                'H': 1,
                'He': 2,
                'Li': 3,
                'Be': 4,
                'B': 5,
                'C': 6,
                'N': 7,
                'O': 8,
                'F': 9,
                'Ne': 10,
                'Na': 11,
                'Mg': 12,
                'Al': 13,
                'Si': 14,
                'P': 15,
                'S': 16,
                'Cl': 17,
                'Ar': 18,
            }
            atomic_nums = [symbol_to_Z[symbol] for symbol in self.model.species_order]
            return AtomicNumberTable(atomic_nums)
        else:
            # Default to HCNO if no species_order is available
            return AtomicNumberTable([1, 6, 7, 8])

    @property
    def atomic_energies(self):
        """
        Get the atomic reference energies.

        Returns
        -------
        List[float] or None
            List of atomic reference energies or None if not available.
        """
        if hasattr(self.model, 'sae_dict'):
            # Return atomic energies if available
            return list(self.model.sae_dict.values())
        elif hasattr(self.model, 'energy_shifter') and hasattr(
            self.model.energy_shifter, 'self_energies'
        ):
            # Return self energies from energy shifter
            return self.model.energy_shifter.self_energies.tolist()
        else:
            # No atomic energies available
            return None

    @property
    def r_max(self):
        """
        Get the maximum cutoff radius.

        Returns
        -------
        float
            Maximum cutoff radius.
        """
        if hasattr(self.model, 'aev_computer'):
            return self.model.aev_computer.Rcr
        else:
            # Default cutoff for ANI-1x
            return 5.2

    @r_max.setter
    def r_max(self, value):
        """
        Set the maximum cutoff radius.

        Parameters
        ----------
        value : float
            New maximum cutoff radius.
        """
        # This is a no-op for ANI models as changing r_max would require
        # rebuilding the AEV computer, which would invalidate the model
        pass
