from __future__ import annotations

import os
from pathlib import Path

import requests
import torch
from tqdm import tqdm

from equitrain.backends.torch_wrappers import MaceWrapper as TorchMaceWrapper


class MaceWrapper(TorchMaceWrapper):
    def __init__(
        self,
        args,
        optimize_atomic_energies=False,
        filename_model: str | os.PathLike[str] | None = None,
        url='https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_epoch-199.model',
    ):
        if filename_model is None:
            filename_model = (
                Path(__file__).resolve().parents[1] / 'tests' / 'mace.model'
            )

        model_path = Path(filename_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if not model_path.exists():
            self._download_model(url, model_path)

        try:
            model = torch.load(model_path, weights_only=False)
        except (RuntimeError, FileNotFoundError):
            # Retry download if the existing archive is corrupted/incomplete
            if model_path.exists():
                model_path.unlink()
            self._download_model(url, model_path)
            model = torch.load(model_path, weights_only=False)

        super().__init__(
            args,
            model,
            optimize_atomic_energies=optimize_atomic_energies,
        )

    @staticmethod
    def _download_model(url: str, filename_model: Path) -> None:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with filename_model.open('wb') as handle:
                for data in tqdm(
                    response.iter_content(chunk_size=8192),
                    desc='Downloading MACE',
                ):
                    if data:
                        handle.write(data)
