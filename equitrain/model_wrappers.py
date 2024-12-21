
import torch

class MaceWrapper(torch.nn.Module):

    def __init__(self, model):

        super().__init__()

        self.model = model


    def forward(self, *args):
        r = self.model(*args, training=self.training)

        if isinstance(r, dict):
            energy = r['energy']
            forces = r['forces']
            stress = r['stress']
        else:
            energy, forces, stress = r

        return energy, forces, stress

