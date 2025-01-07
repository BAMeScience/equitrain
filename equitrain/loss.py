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
    def __init__(self, angle_weight=1.0, epsilon=1e-8):
        super().__init__()

        self.angle_weight = angle_weight
        self.epsilon = epsilon

    def forward(self, input, target, weights=None):
        # Compute lengths of force vectors
        n1 = torch.norm(target, dim=1)
        n2 = torch.norm(input, dim=1)
        # Compute angle between force vectors
        angle = self.compute_angle(target, input, n1=n1, n2=n2)

        # Loss is the sum of normalized length mismath and angle discrepancy
        return torch.mean(
            *(
                torch.abs(n1 - n2) / (0.5 * n1 + 0.5 * n2 + self.epsilon)
                + self.angle_weight * angle
            )
        )

    def compute_angle(self, s, t, n1=None, n2=None):
        if n1 is None:
            n1 = torch.norm(s, dim=1)
        if n2 is None:
            n2 = torch.norm(t, dim=1)
        # Compute dot product between force vectors
        dp = torch.einsum('ij,ij->i', s, t)
        # Compute angle, use tanh for numerical stability
        return torch.arccos(dp / (n1 * n2 + self.epsilon))


class LossComponent:
    def __init__(self, value: torch.Tensor = None, n: torch.Tensor = None, device=None):
        self.value = value
        self.n = n

        if value is None:
            self.value = torch.tensor(0.0, device=device)

        if n is None:
            self.n = torch.tensor(0.0, device=device)

    def __iadd__(self, component: 'LossComponent'):
        self.value = (self.value * self.n + component.value * component.n) / (
            self.n + component.n
        )
        self.n += component.n

        return self

    def detach(self):
        r = LossComponent()

        r.value = self.value.detach()
        r.n = self.n.detach()

        return r

    def gather_for_metrics(self, accelerator):
        r = LossComponent(device=accelerator.device)

        values = accelerator.gather_for_metrics(self.value.detach())
        ns = accelerator.gather_for_metrics(self.n.detach())

        if len(values.shape) == 0:
            # Single processing context
            r += LossComponent(value=values, n=ns)

        else:
            # Multi-processing context
            assert len(values) == len(ns)

            for i in range(len(values)):
                r += LossComponent(value=values[i], n=ns[i])

        return r


class Loss(dict):
    def __init__(self, device=None):
        self['total'] = LossComponent(device=device)
        self['energy'] = LossComponent(device=device)
        self['forces'] = LossComponent(device=device)
        self['stress'] = LossComponent(device=device)

    def __iadd__(self, loss: 'Loss'):
        for key, component in loss.items():
            self[key] += component

        return self

    def isnan(self):
        return torch.isnan(self['total'].value)

    def detach(self):
        r = Loss()

        for key, value in self.items():
            r[key] = value.detach()

        return r

    def gather_for_metrics(self, accelerator):
        result = Loss(device=accelerator.device)

        for key, component in self.items():
            result[key] = component.gather_for_metrics(accelerator)

        return result


class GenericLossFn(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        stress_weight=0.0,
        # As opposed to forces, energy is predicted per material. By normalizing
        # the energy by the number of atoms, forces and energy become comparable
        loss_energy_per_atom=True,
        **args,
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
        if energy_loss is not None and (
            not math.isinf(energy_loss) or self.energy_weight > 0.0
        ):
            result += self.energy_weight * energy_loss
        if forces_loss is not None and (
            not math.isinf(forces_loss) or self.forces_weight > 0.0
        ):
            result += self.forces_weight * forces_loss
        if stress_loss is not None and (
            not math.isinf(stress_loss) or self.stress_weight > 0.0
        ):
            result += self.stress_weight * stress_loss

        return result

    def forward(self, y_pred, y_true):
        loss = Loss(device=y_true.batch.device)

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

        # Evaluate every loss component
        if self.energy_weight > 0.0:
            loss_e = self.loss_energy(e_pred, e_true, weights=energy_weights)
        if self.forces_weight > 0.0:
            loss_f = self.loss_forces(f_pred, f_true)
        if self.stress_weight > 0.0:
            loss_s = self.loss_stress(s_pred, s_true)

        # Move results to loss object
        loss['total'].value += self.compute_weighted_loss(loss_e, loss_f, loss_s)
        loss['total'].n += y_true.batch.max() + 1

        if self.energy_weight > 0.0:
            loss['energy'].value += loss_e
            loss['energy'].n += e_true.numel()
        if self.forces_weight > 0.0:
            loss['forces'].value += loss_f
            loss['forces'].n += f_true.numel()
        if self.stress_weight > 0.0:
            loss['stress'].value += loss_s
            loss['stress'].n += s_true.numel()

        return loss
