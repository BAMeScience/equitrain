
import torch

class MaceWrapper(torch.nn.Module):

    def __init__(self, args, model):

        super().__init__()

        self.model = model
        self.compute_force  = args.forces_weight > 0.0
        self.compute_stress = args.stress_weight > 0.0


    def forward(self, *args):

        y_pred = self.model(*args, compute_force = self.compute_force, compute_stress = self.compute_stress, training=self.training)

        if not isinstance(y_pred, dict):
            y_pred = {'energy': y_pred[0], 'forces': y_pred[1], 'stress': y_pred[2]}

        return y_pred
