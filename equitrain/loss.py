import math
import torch


class WeightedL1Loss(torch.nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, input, target, weights=None):
        """
        Compute the weighted L1 loss.

        Args:
            input (torch.Tensor): Predicted values.
            target (torch.Tensor): Ground truth values.
            weights (torch.Tensor): Weights for each element.

        Returns:
            torch.Tensor: Weighted L1 loss.
        """
        loss = torch.abs(input - target)

        if weights is not None:
            loss *= weights

        return loss.mean()


class ForceAngleLoss(torch.nn.Module):

    def __init__(self, angle_weight = 1.0, epsilon = 1e-8):
        super().__init__()

        self.angle_weight = angle_weight
        self.epsilon      = epsilon


    def forward(self, input, target, weights=None):
        # Compute lengths of force vectors
        n1 = torch.norm(target, dim=1)
        n2 = torch.norm(input , dim=1)
        # Compute angle between force vectors
        angle = compute_angle(target, input, n1=n1, n2=n2)

        # Loss is the sum of normalized length mismath and angle discrepancy
        return torch.mean(
            * (torch.abs(n1 - n2)/(0.5*n1 + 0.5*n2 + self.epsilon) + self.angle_weight*angle)
        )


    def compute_angle(self, s, t, n1 = None, n2 = None):
        if n1 is None:
            n1 = torch.norm(s, dim=1)
        if n2 is None:
            n2 = torch.norm(t, dim=1)
        # Compute dot product between force vectors
        dp = torch.einsum('ij,ij->i', s, t)
        # Compute angle, use tanh for numerical stability
        return torch.arccos(dp / (n1*n2 + self.epsilon))


class Loss(dict):

    def __init__(
        self,
        total : torch.Tensor = None,
        energy: torch.Tensor = None,
        forces: torch.Tensor = None,
        stress: torch.Tensor = None,
        n     : torch.Tensor = None,
        device = None
        ):

        self['total' ] = total  if total  is not None else torch.tensor(0.0, device=device)
        self['energy'] = energy if energy is not None else torch.tensor(0.0, device=device)
        self['forces'] = forces if forces is not None else torch.tensor(0.0, device=device)
        self['stress'] = stress if stress is not None else torch.tensor(0.0, device=device)

        self.n = n if n is not None else torch.tensor(0, device=device, requires_grad=False)


    def __iadd__(self, loss : "Loss"):

        for key, value in loss.items():
            self[key] += value

        self.n += loss.n

        return self


    def __itruediv__(self, x):

        for key in self:
            self[key] /= x

        return self


    def isnan(self):
        return torch.isnan(self['total'])


    def detach(self):

        r = Loss()

        for key, value in self.items():

            r[key] = value.detach()

        r.n = self.n.detach()

        return r


    def gather_for_metrics(self, accelerator, reduction="mean"):

        result = Loss(device = accelerator.device)

        for key, value in self.items():
            result[key] = accelerator.gather_for_metrics(value.detach()).mean()

        result.n = accelerator.gather_for_metrics(self.n.detach()).sum()

        return result


class GenericLossFn(torch.nn.Module):

    def __init__(
        self,
        energy_weight = 1.0,
        forces_weight = 1.0,
        stress_weight = 0.0,
        # As opposed to forces, energy is predicted per material. By normalizing
        # the energy by the number of atoms, forces and energy become comparable
        loss_energy_per_atom = True,
        **args
        ):

        super().__init__()

        # TODO: Allow to select other loss functions with args
        self.loss_energy = WeightedL1Loss()
        self.loss_forces = torch.nn.L1Loss()
        self.loss_stress = torch.nn.L1Loss()

        # TODO: Use register_buffer instead
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.stress_weight = stress_weight

        self.loss_energy_per_atom = loss_energy_per_atom


    def compute_weighted_loss(self, energy_loss, forces_loss, stress_loss):
        result = 0.0
        # handle initial values correctly when weights are zero, i.e. 0.0*Inf -> NaN
        if energy_loss is not None and (not math.isinf(energy_loss) or self.energy_weight > 0.0):
            result += self.energy_weight * energy_loss
        if forces_loss is not None and (not math.isinf(forces_loss) or self.forces_weight > 0.0):
            result += self.forces_weight * forces_loss
        if stress_loss is not None and (not math.isinf(stress_loss) or self.stress_weight > 0.0):
            result += self.stress_weight * stress_loss

        return result


    def forward(self, y_pred, y_true):

        energy_weights = None

        if self.loss_energy_per_atom:
            num_atoms = y_true.ptr[1:] - y_true.ptr[:-1]
            energy_weights = 1.0 / num_atoms

        e_true = y_true.y
        f_true = y_true['force']
        s_true = y_true['stress']

        e_pred = y_pred['energy']
        f_pred = y_pred['forces']
        s_pred = y_pred['stress']

        loss_e = None
        loss_f = None
        loss_s = None

        if self.energy_weight > 0.0:
            loss_e = self.loss_energy(e_pred, e_true, weights=energy_weights)
        if self.forces_weight > 0.0:
            loss_f = self.loss_forces(f_pred, f_true)
        if self.stress_weight > 0.0:
            loss_s = self.loss_stress(s_pred, s_true)

        loss = self.compute_weighted_loss(loss_e, loss_f, loss_s)
        n    = torch.tensor(y_true.natoms.shape[0], device=y_true.natoms.device, requires_grad = False)

        return Loss(loss, energy = loss_e, forces = loss_f, stress = loss_s, n = n)
