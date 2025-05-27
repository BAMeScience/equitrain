"""
Utility test wrapper for ORB models.

This module provides test utilities for ORB (Orbital Materials) models
within the Equitrain framework.
"""

import torch

from equitrain.model_wrappers import OrbWrapper


class OrbWrapper(OrbWrapper):
    """
    Test utility wrapper for ORB models.

    This class extends the OrbWrapper to provide functionality for testing,
    including easy model creation and validation utilities.
    """

    def __init__(
        self,
        args,
        model=None,
        model_variant='direct',
        enable_zbl=False,
        model_size='small',
    ):
        """
        Initialize the test ORB wrapper.

        Parameters
        ----------
        args : object
            Arguments object containing training parameters
        model : torch.nn.Module, optional
            A pre-trained ORB model. If None, a new model will be created.
        model_variant : str, optional
            ORB model variant ('direct' or 'conservative'). Default is 'direct'.
        enable_zbl : bool, optional
            Enable ZBL repulsion term for high-Z elements. Default is False.
        model_size : str, optional
            Size of the model to load ('small', 'medium', 'large'). Default is 'small'.
        """
        self.model_size = model_size
        super().__init__(args, model, model_variant, enable_zbl)

    def _create_default_model(self):
        """
        Create a default ORB model from the model zoo.

        Returns
        -------
        torch.nn.Module
            An ORB model from the OMat24 model zoo
        """
        try:
            from orb_models.forcefield import pretrained

            # Select model based on size and variant
            if self.model_variant == 'direct':
                if self.model_size == 'small':
                    model = pretrained.orb_v3_small_direct()
                elif self.model_size == 'medium':
                    model = pretrained.orb_v3_medium_direct()
                elif self.model_size == 'large':
                    model = pretrained.orb_v3_large_direct()
                else:
                    model = pretrained.orb_v3_small_direct()
            else:
                if self.model_size == 'small':
                    model = pretrained.orb_v3_small()
                elif self.model_size == 'medium':
                    model = pretrained.orb_v3_medium()
                elif self.model_size == 'large':
                    model = pretrained.orb_v3_large()
                else:
                    model = pretrained.orb_v3_small()

            # Enable ZBL repulsion if requested
            if self.enable_zbl:
                model.enable_zbl = True

            return model
        except ImportError:
            raise ImportError(
                'ORB models are required. Install with: pip install "orb-models>=3.0"'
            )

    @classmethod
    def get_initial_model(
        cls, model_variant='direct', model_size='small', enable_zbl=False
    ):
        """
        Get an initial ORB model for testing.

        Parameters
        ----------
        model_variant : str, optional
            ORB model variant ('direct' or 'conservative'). Default is 'direct'.
        model_size : str, optional
            Size of the model ('small', 'medium', 'large'). Default is 'small'.
        enable_zbl : bool, optional
            Enable ZBL repulsion term. Default is False.

        Returns
        -------
        torch.nn.Module
            An initialized ORB model.
        """
        try:
            from orb_models.forcefield import pretrained

            # Select model based on parameters
            if model_variant == 'direct':
                if model_size == 'small':
                    model = pretrained.orb_v3_small_direct()
                elif model_size == 'medium':
                    model = pretrained.orb_v3_medium_direct()
                elif model_size == 'large':
                    model = pretrained.orb_v3_large_direct()
                else:
                    model = pretrained.orb_v3_small_direct()
            else:
                if model_size == 'small':
                    model = pretrained.orb_v3_small()
                elif model_size == 'medium':
                    model = pretrained.orb_v3_medium()
                elif model_size == 'large':
                    model = pretrained.orb_v3_large()
                else:
                    model = pretrained.orb_v3_small()

            # Enable ZBL repulsion if requested
            if enable_zbl:
                model.enable_zbl = True

            return model
        except ImportError:
            raise ImportError(
                'ORB models are required. Install with: pip install "orb-models>=3.0"'
            )

    def validate_matbench_mae(self, energy_pred, energy_true, forces_pred, forces_true):
        """
        Compute Matbench-style MAE for energy and forces.

        Parameters
        ----------
        energy_pred : torch.Tensor
            Predicted energies
        energy_true : torch.Tensor
            True energies
        forces_pred : torch.Tensor
            Predicted forces
        forces_true : torch.Tensor
            True forces

        Returns
        -------
        dict
            Dictionary containing energy and force MAE values
        """
        # Energy MAE (eV/atom)
        energy_mae = torch.mean(torch.abs(energy_pred - energy_true)).item()

        # Force MAE (eV/Å)
        forces_mae = torch.mean(torch.abs(forces_pred - forces_true)).item()

        return {'energy_mae_eV_per_atom': energy_mae, 'forces_mae_eV_per_A': forces_mae}

    def get_confidence_scores(self, *args):
        """
        Get confidence scores from ORB model if available.

        Parameters
        ----------
        *args : tuple
            Input arguments for the model

        Returns
        -------
        dict or None
            Confidence scores if available, None otherwise
        """
        # Check if model has confidence head
        if (
            hasattr(self.model, 'confidence_head')
            and self.model.confidence_head is not None
        ):
            try:
                with torch.no_grad():
                    result = self.forward(*args)
                    if 'confidence' in result:
                        return result['confidence']
            except Exception:
                pass

        return None
