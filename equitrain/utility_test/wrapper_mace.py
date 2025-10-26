import os

import requests
import torch
from tqdm import tqdm

from equitrain.model_wrappers import MaceWrapper


class MaceWrapper(MaceWrapper):
    def __init__(
        self,
        args,
        optimize_atomic_energies=False,
        filename_model='mace.model',
        url='https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_epoch-199.model',
    ):
        if not os.path.exists(filename_model):
            self._download_model(url, filename_model)

        try:
            model = torch.load(filename_model, weights_only=False)
        except (RuntimeError, FileNotFoundError):
            # Retry download if the existing archive is corrupted/incomplete
            if os.path.exists(filename_model):
                os.remove(filename_model)
            self._download_model(url, filename_model)
            model = torch.load(filename_model, weights_only=False)

        super().__init__(
            args,
            model,
            optimize_atomic_energies=optimize_atomic_energies,
        )

    @staticmethod
    def _download_model(url: str, filename_model: str) -> None:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(filename_model, 'wb') as handle:
                for data in tqdm(
                    response.iter_content(chunk_size=8192),
                    desc='Downloading MACE',
                ):
                    if data:
                        handle.write(data)
