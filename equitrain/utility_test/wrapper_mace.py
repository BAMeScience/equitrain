import os
import requests
import torch

from tqdm import tqdm


class MaceWrapper(torch.nn.Module):

    def __init__(self, filename_model = "mace.model", url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_epoch-199.model"):

        super().__init__()

        if not os.path.exists(filename_model):
            
            response = requests.get(url, stream=True)

            with open(filename_model, "wb") as handle:
                for data in tqdm(response.iter_content(), desc="Downloading MACE"):
                    handle.write(data)

        self.model = torch.load(filename_model)


    def forward(self, *args):

        y_pred = self.model(*args, training=self.training)

        if not isinstance(y_pred, dict):
            y_pred = {'energy': y_pred[0], 'forces': y_pred[1], 'stress': y_pred[2]}

        return y_pred
