import os
import requests
import torch

from tqdm import tqdm

from equitrain.model_wrappers import MaceWrapper

class MaceWrapper(MaceWrapper):

    def __init__(self, args, filename_model = "mace.model", url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_epoch-199.model"):

        if not os.path.exists(filename_model):
            
            response = requests.get(url, stream=True)

            with open(filename_model, "wb") as handle:
                for data in tqdm(response.iter_content(), desc="Downloading MACE"):
                    handle.write(data)

        super().__init__(args, torch.load(filename_model))
