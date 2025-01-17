import torch

from equitrain.data.scatter import scatter_mean


class L1LossEnergy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, weights):
        # TODO: Different loss types can be implemented here
        error = torch.abs(input - target)
        error *= weights

        loss = error.mean()
        error = error.detach()

        return loss, error


class L1LossForces(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, batch):
        # TODO: Different loss types can be implemented here
        error = torch.abs(input - target)

        loss = error.mean()

        error = error.detach()
        error = error.mean(dim=1)
        error = scatter_mean(error, batch)

        return loss, error


class L1LossStress(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # TODO: Different loss types can be implemented here
        error = torch.abs(input - target)

        loss = error.mean()
        error = error.detach()
        error = error.mean(dim=(1, 2))

        return loss, error


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
        skip = (ns == 0.0).any().item()

        if len(values.shape) == 0:
            # Single processing context
            r += LossComponent(value=values, n=ns)

        else:
            # Multi-processing context
            assert len(values) == len(ns)

            for i in range(len(values)):
                r += LossComponent(value=values[i], n=ns[i])

        return r, skip


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
        skip = {}

        for key, component in self.items():
            result[key], skip[key] = component.gather_for_metrics(accelerator)

        return result, skip


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
        self.loss_energy = L1LossEnergy()
        self.loss_forces = L1LossForces()
        self.loss_stress = L1LossStress()

        # TODO: Use register_buffer instead
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.stress_weight = stress_weight

        self.loss_energy_per_atom = loss_energy_per_atom

    def compute_weighted(self, energy_value, forces_value, stress_value):
        result = 0.0
        # handle initial values correctly when weights are zero, i.e. 0.0*Inf -> NaN
        if energy_value is not None and (
            not torch.isinf(energy_value).any() or self.energy_weight > 0.0
        ):
            result += self.energy_weight * energy_value
        if forces_value is not None and (
            not torch.isinf(forces_value).any() or self.forces_weight > 0.0
        ):
            result += self.forces_weight * forces_value
        if stress_value is not None and (
            not torch.isinf(stress_value).any() or self.stress_weight > 0.0
        ):
            result += self.stress_weight * stress_value

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

        error_e = None
        error_f = None
        error_s = None

        # Evaluate every loss component
        if self.energy_weight > 0.0:
            loss_e, error_e = self.loss_energy(e_pred, e_true, energy_weights)
        if self.forces_weight > 0.0:
            loss_f, error_f = self.loss_forces(f_pred, f_true, y_true.batch)
        if self.stress_weight > 0.0:
            loss_s, error_s = self.loss_stress(s_pred, s_true)

        # Move results to loss object
        loss['total'].value += self.compute_weighted(loss_e, loss_f, loss_s)
        loss['total'].n += y_true.batch.max() + 1

        if self.energy_weight > 0.0:
            loss['energy'].value = loss_e
            loss['energy'].n += e_true.numel()
        if self.forces_weight > 0.0:
            loss['forces'].value += loss_f
            loss['forces'].n += f_true.numel()
        if self.stress_weight > 0.0:
            loss['stress'].value += loss_s
            loss['stress'].n += s_true.numel()

        error = self.compute_weighted(error_e, error_f, error_s)

        return loss, error
