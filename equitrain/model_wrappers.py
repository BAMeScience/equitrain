
import torch

class MaceWrapper(torch.nn.Module):

    def __init__(self, args, model):

        super().__init__()

        self.model = model
        self.compute_force  = args.force_weight  > 0.0
        self.compute_stress = args.stress_weight > 0.0


    def forward(self, *args):

        r = self.model(*args, compute_force = self.compute_force, compute_stress = self.compute_stress, training=self.training)

        if isinstance(r, dict):
            energy = r['energy']
            forces = r['forces']
            stress = r['stress']
        else:
            energy, forces, stress = r

        return energy, forces, stress
