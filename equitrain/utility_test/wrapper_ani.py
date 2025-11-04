"""
Utility test wrapper for ANI models.

This module provides a utility wrapper for ANI models to be used in tests.
"""

import os

import torch
import torchani

from equitrain.model_wrappers import AniWrapper


class AniWrapper(AniWrapper):
    """
    Utility wrapper for ANI models to be used in tests.

    This wrapper extends the AniWrapper class to provide additional functionality
    for testing, such as downloading pre-trained models.

    Parameters
    ----------
    args : object
        Arguments object containing training parameters
    model_type : str, optional
        Type of ANI model to use. Options are 'ani1x', 'ani1ccx', or 'ani2x'.
        Default is 'ani1x'.
    """

    def __init__(
        self,
        args,
        model_type='ani1x',
        filename_model=None,
    ):
        # Determine which model to use
        if model_type == 'ani1x':
            model_class = torchani.models.ANI1x
            default_filename = 'ani1x.pt'
        elif model_type == 'ani1ccx':
            model_class = torchani.models.ANI1ccx
            default_filename = 'ani1ccx.pt'
        elif model_type == 'ani2x':
            model_class = torchani.models.ANI2x
            default_filename = 'ani2x.pt'
        else:
            raise ValueError(f'Unknown model type: {model_type}')

        # Use provided filename or default
        if filename_model is None:
            filename_model = default_filename

        # Load or download the model
        if not os.path.exists(filename_model):
            # Create a new model instance
            model = model_class()
            # Save the model to disk
            torch.save(model, filename_model)
            print(f'Created and saved new {model_type} model to {filename_model}')
        else:
            # Load the model from disk
            model = torch.load(filename_model, weights_only=False)
            print(f'Loaded {model_type} model from {filename_model}')

        # Set species order based on model type
        if model_type == 'ani1x' or model_type == 'ani1ccx':
            species_order = ['H', 'C', 'N', 'O']
        elif model_type == 'ani2x':
            species_order = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']

        # Initialize the parent class
        super().__init__(args, model, species_order=species_order)
