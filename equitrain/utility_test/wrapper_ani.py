"""
Utility test wrapper for ANI models.

This module provides a utility wrapper for ANI models to be used in tests.
"""

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
        elif model_type == 'ani1ccx':
            model_class = torchani.models.ANI1ccx
        elif model_type == 'ani2x':
            model_class = torchani.models.ANI2x
        else:
            raise ValueError(f'Unknown model type: {model_type}')

        if filename_model is not None:
            model = torchani.utils.load(filename_model)
        else:
            model = model_class()

        if model_type in {'ani1x', 'ani1ccx'}:
            species_order = ['H', 'C', 'N', 'O']
        elif model_type == 'ani2x':
            species_order = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']

        # Initialize the parent class
        super().__init__(args, model, species_order=species_order)
