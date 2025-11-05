from __future__ import annotations

import copy

import torch

from equitrain.data.backend_torch.scatter import scatter_mean

from .torch_loss import Loss, LossCollection


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

        if loss_type is None or loss_type in {'mae', 'l1'}:
            self.error_fn = lambda x, y: torch.nn.functional.l1_loss(
                x, y, reduction='none'
            )
        elif loss_type in {'smooth-mae', 'smooth-l1'}:
            self.error_fn = lambda x, y: torch.nn.functional.smooth_l1_loss(
                x, y, beta=smooth_l1_beta, reduction='none'
            )
        elif loss_type in {'mse', 'l2'}:
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
        error = self.error_fn(input, target)
        if self.loss_clipping is not None:
            error = torch.clamp(error, max=self.loss_clipping)

        if self.weight_fn is not None:
            weights = self.weight_fn(input, target)
            if error.ndim > 1:
                error = weights[:, None] * error
            else:
                error = weights * error

        return error

    @staticmethod
    def rowwise_norm(x):
        return x.norm(dim=-1) if x.ndim > 1 else x.abs()


class LossFnEnergy(torch.nn.Module):
    def __init__(self, **args):
        super().__init__()
        args = copy.deepcopy(args)
        if args.get('loss_type_energy') is not None:
            args['loss_type'] = args['loss_type_energy']
        if args.get('loss_weight_type_energy') is not None:
            args['loss_weight_type'] = args['loss_weight_type_energy']
        self.error_fn = ErrorFn(**args)

    def forward(self, input, target, weights):
        error = self.error_fn(input, target)
        if weights is not None:
            error = error * weights
        loss = error.mean()
        return loss, error.detach()


class LossFnForces(torch.nn.Module):
    def __init__(self, **args):
        super().__init__()
        if args.get('loss_type_forces') is not None:
            args['loss_type'] = args['loss_type_forces']
        if args.get('loss_weight_type_forces') is not None:
            args['loss_weight_type'] = args['loss_weight_type_forces']
        self.error_fn = ErrorFn(**args)

    def forward(self, input, target, batch):
        error = self.error_fn(input, target)
        loss = error.mean()
        error = scatter_mean(error.detach().mean(dim=1), batch)
        return loss, error


class LossFnStress(torch.nn.Module):
    def __init__(self, **args):
        super().__init__()
        if args.get('loss_type_stress') is not None:
            args['loss_type'] = args['loss_type_stress']
        if args.get('loss_weight_type_stress') is not None:
            args['loss_weight_type'] = args['loss_weight_type_stress']
        self.error_fn = ErrorFn(**args)

    def forward(self, input, target):
        error = self.error_fn(input, target)
        loss = error.mean()
        return loss, error.detach().mean(dim=(1, 2))


class LossFn(torch.nn.Module):
    def __init__(
        self,
        energy_weight: float = 1.0,
        forces_weight: float = 1.0,
        stress_weight: float = 0.0,
        loss_energy_per_atom: bool = True,
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
            energy_weights = (1.0 / num_atoms).to(dtype=y_pred['energy'].dtype, device=y_pred['energy'].device)

        e_true = y_true.y
        f_true = y_true['force']
        s_true = y_true['stress']

        e_pred = y_pred['energy']
        f_pred = y_pred['forces']
        s_pred = y_pred['stress']

        loss_e = loss_f = loss_s = None
        error_e = error_f = error_s = None

        if self.energy_weight > 0.0:
            loss_e, error_e = self.loss_energy(e_pred, e_true, energy_weights)
        if self.forces_weight > 0.0:
            loss_f, error_f = self.loss_forces(f_pred, f_true, y_true.batch)
        if self.stress_weight > 0.0:
            loss_s, error_s = self.loss_stress(s_pred, s_true)

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
        if not isinstance(error, torch.Tensor):
            error = torch.tensor(error, device=e_pred.device, dtype=e_pred.dtype)
        else:
            error = error.to(dtype=e_pred.dtype, device=e_pred.device)
        return loss, error


class LossFnCollection(torch.nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.main = LossFn(**args)
        self.loss_fns = {}
        for loss_type in args['loss_monitor']:
            args_for_type = {**args, 'loss_type': loss_type}
            self.loss_fns[loss_type] = LossFn(**args_for_type)

    def forward(self, y_pred, y_true):
        loss = LossCollection(list(self.loss_fns.keys()), device=y_true.batch.device)
        loss.main, error = self.main(y_pred, y_true)
        for loss_type, loss_fn in self.loss_fns.items():
            detached_pred = {
                key: value.detach() if value is not None else None
                for key, value in y_pred.items()
            }
            loss[loss_type], _ = loss_fn(detached_pred, y_true)
        return loss, error


__all__ = [
    'ErrorFn',
    'LossFnEnergy',
    'LossFnForces',
    'LossFnStress',
    'LossFn',
    'LossFnCollection',
]
