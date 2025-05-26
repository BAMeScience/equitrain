import os

import matgl
import requests
import torch
from tqdm import tqdm

from equitrain.model_wrappers import M3GNetWrapper


class M3GNetWrapper(M3GNetWrapper):
    """
    Test utility wrapper for M3GNet models.

    This class extends the M3GNetWrapper to provide functionality for testing,
    including downloading pre-trained models if needed.
    """

    def __init__(
        self,
        args,
        filename_model='m3gnet.pt',
        url='https://github.com/materialsvirtuallab/matgl/releases/download/v0.5.0/M3GNet-MP-2021.2.8-PES.pt',
        element_types=None,
    ):
        """
        Initialize the test M3GNet wrapper.

        Parameters
        ----------
        args : object
            Arguments object containing training parameters
        filename_model : str, optional
            Filename to save the downloaded model. Default is 'm3gnet.pt'.
        url : str, optional
            URL to download the pre-trained model from. Default is the MatGL M3GNet universal potential.
        element_types : list[str], optional
            List of chemical symbols in order. Default is None, which will use the model's
            element_types if available.
        """
        # Download the model if it doesn't exist
        if not os.path.exists(filename_model):
            response = requests.get(url, stream=True)

            with open(filename_model, 'wb') as handle:
                for data in tqdm(response.iter_content(), desc='Downloading M3GNet'):
                    handle.write(data)

        # Load the model using MatGL's load_model function
        model = matgl.load_model(filename_model)

        # Initialize the parent class
        super().__init__(args, model, element_types=element_types)

    @classmethod
    def get_initial_model(cls, element_types=None):
        """
        Get an initial M3GNet model.

        Parameters
        ----------
        element_types : list[str], optional
            List of chemical symbols to include in the model. Default is None,
            which will use a default set of elements.

        Returns
        -------
        torch.nn.Module
            An initialized M3GNet model.
        """
        # If no element_types provided, use a default set covering common elements
        if element_types is None:
            element_types = [
                'H',
                'He',
                'Li',
                'Be',
                'B',
                'C',
                'N',
                'O',
                'F',
                'Ne',
                'Na',
                'Mg',
                'Al',
                'Si',
                'P',
                'S',
                'Cl',
                'Ar',
                'K',
                'Ca',
                'Sc',
                'Ti',
                'V',
                'Cr',
                'Mn',
                'Fe',
                'Co',
                'Ni',
                'Cu',
                'Zn',
                'Ga',
                'Ge',
                'As',
                'Se',
                'Br',
                'Kr',
                'Rb',
                'Sr',
                'Y',
                'Zr',
                'Nb',
                'Mo',
                'Tc',
                'Ru',
                'Rh',
                'Pd',
                'Ag',
                'Cd',
                'In',
                'Sn',
                'Sb',
                'Te',
                'I',
                'Xe',
                'Cs',
                'Ba',
                'La',
                'Ce',
                'Hf',
                'Ta',
                'W',
                'Re',
                'Os',
                'Ir',
                'Pt',
                'Au',
                'Hg',
                'Tl',
                'Pb',
                'Bi',
            ]

        # Create a new M3GNet model with the specified element_types
        from matgl.models import M3GNet

        model = M3GNet(
            element_types=tuple(element_types),
            cutoff=5.0,
            threebody_cutoff=4.0,
            nblocks=3,
            max_n=3,
            max_l=3,
            units=64,
            ntargets=1,
        )

        return model
