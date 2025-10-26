import copy

import torch

from equitrain.data.backend_torch.scatter import scatter_mean
from equitrain.loss import Loss, LossCollection


class ErrorFn(torch.nn.Module):
    def __init__(
        self,
        loss_type: str = None,
        loss_weight_type: str = None,
        smooth_l1_beta: float = None,
        huber_delta: float = None,
        loss_clipping: float = None,
        **args,
    ):
        super().__init__()

        loss_type = loss_type.lower() if loss_type else None
        loss_weight_type = loss_weight_type.lower() if loss_weight_type else None

        if loss_type is None or loss_type == 'mae' or loss_type == 'l1':
            self.error_fn = lambda x, y: torch.nn.functional.l1_loss(
                x, y, reduction='none'
            )

        elif loss_type == 'smooth-mae' or loss_type == 'smooth-l1':
            self.error_fn = lambda x, y: torch.nn.functional.smooth_l1_loss(
                x, y, beta=smooth_l1_beta, reduction='none'
            )

        elif loss_type == 'mse' or loss_type == 'l2':
            self.error_fn = lambda x, y: torch.nn.functional.mse_loss(
                x, y, reduction='none'
            )

        elif loss_type == 'huber':
            self.error_fn = lambda x, y: torch.nn.functional.huber_loss(
                x, y, delta=huber_delta, reduction='none'
            )

        else:
            raise ValueError(f'Invalid loss type: {loss_type}')

        if loss_weight_type == 'groundstate':
            self.weight_fn = (
                lambda x, y: torch.exp(-1000.0 * self.rowwise_norm(y) ** 2) + 1.0
            )

        else:
            self.weight_fn = None

        self.loss_clipping = loss_clipping

    def forward(self, input, target):
        x = self.error_fn(input, target)
        # Clamp loss to avoid exploding gradients
        if self.loss_clipping is not None:
            x = torch.clamp(x, max=self.loss_clipping)

        if self.weight_fn is not None:
            weights = self.weight_fn(input, target)
            if x.ndim > 1:
                x = weights[:, None] * x
            else:
                x = weights * x

        return x

    @staticmethod
    def rowwise_norm(x):
        return x.norm(dim=-1) if x.ndim > 1 else x.abs()


class LossFnEnergy(torch.nn.Module):
    def __init__(self, **args):
        super().__init__()

        args = copy.deepcopy(args)

        if 'loss_type_energy' in args and args['loss_type_energy'] is not None:
            args['loss_type'] = args['loss_type_energy']

        if (
            'loss_weight_type_energy' in args
            and args['loss_weight_type_energy'] is not None
        ):
            args['loss_weight_type'] = args['loss_weight_type_energy']

        self.error_fn = ErrorFn(**args)

    def forward(self, input, target, weights):
        error = self.error_fn(input, target)
        error *= weights

        loss = error.mean()
        error = error.detach()

        return loss, error


class LossFnForces(torch.nn.Module):
    def __init__(self, **args):
        super().__init__()

        if 'loss_type_forces' in args and args['loss_type_forces'] is not None:
            args['loss_type'] = args['loss_type_forces']

        if (
            'loss_weight_type_forces' in args
            and args['loss_weight_type_forces'] is not None
        ):
            args['loss_weight_type'] = args['loss_weight_type_forces']

        self.error_fn = ErrorFn(**args)

    def forward(self, input, target, batch):
        error = self.error_fn(input, target)

        loss = error.mean()

        error = error.detach()
        error = error.mean(dim=1)
        error = scatter_mean(error, batch)

        return loss, error


class LossFnStress(torch.nn.Module):
    def __init__(self, **args):
        super().__init__()

        if 'loss_type_stress' in args and args['loss_type_stress'] is not None:
            args['loss_type'] = args['loss_type_stress']

        if (
            'loss_weight_type_stress' in args
            and args['loss_weight_type_stress'] is not None
        ):
            args['loss_weight_type'] = args['loss_weight_type_stress']

        self.error_fn = ErrorFn(**args)

    def forward(self, input, target):
        error = self.error_fn(input, target)

        loss = error.mean()
        error = error.detach()
        error = error.mean(dim=(1, 2))

        return loss, error


class LossFnForcesAngle(torch.nn.Module):
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


class LossFn(torch.nn.Module):
    def __init__(
        self,
        energy_weight: float = 1.0,
        forces_weight: float = 1.0,
        stress_weight: float = 0.0,
        # As opposed to forces, energy is predicted per material. By normalizing
        # the energy by the number of atoms, forces and energy become comparable
        loss_energy_per_atom: bool = True,
        # Loss function arguments
        **args,
    ):
        super().__init__()

        self.loss_energy = LossFnEnergy(**args)
        self.loss_forces = LossFnForces(**args)
        self.loss_stress = LossFnStress(**args)

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


class LossFnCollection(torch.nn.Module):
    def __init__(self, **args):
        super().__init__()

        # Main loss function
        self.main = LossFn(**args)

        # Additional loss metrics
        self.loss_fns = {}
        for loss_type in args['loss_monitor']:
            args_new = {**args, 'loss_type': loss_type}
            self.loss_fns[loss_type] = LossFn(**args_new)

    def forward(self, y_pred, y_true):
        loss = LossCollection(list(self.loss_fns.keys()), device=y_true.batch.device)
        # Evaluate main loss function
        loss.main, error = self.main(y_pred, y_true)

        # Evaluate additional loss metrics
        for loss_type, loss_fn in self.loss_fns.items():
            loss[loss_type], _ = loss_fn(
                # Detach predictions for other loss functions
                {
                    key: value.detach() if value is not None else None
                    for key, value in y_pred.items()
                },
                y_true,
            )

        return loss, error
