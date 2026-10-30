from __future__ import annotations

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
            self.weight_fn = lambda x, y: (
                torch.exp(-1000.0 * self.rowwise_norm(y) ** 2) + 1.0
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


def _weighted_error(error_fn, input, target, weights=None):
    if weights is None:
        error = error_fn(input, target)
        return error, error.mean()
    weights = torch.broadcast_to(weights.to(input), input.shape)
    valid = weights != 0
    # Mask before computing the error so missing targets cannot introduce NaNs.
    error = (
        error_fn(torch.where(valid, input, 0.0), torch.where(valid, target, 0.0))
        * weights
    )
    return error, error.sum() / valid.sum().clamp_min(1)


class LossFnEnergy(torch.nn.Module):
    def __init__(self, **args):
        super().__init__()
        args = dict(args)
        if args.get('loss_type_energy') is not None:
            args['loss_type'] = args['loss_type_energy']
        if args.get('loss_weight_type_energy') is not None:
            args['loss_weight_type'] = args['loss_weight_type_energy']
        self.error_fn = ErrorFn(**args)

    def forward(self, input, target, weights):
        error, loss = _weighted_error(self.error_fn, input, target, weights)
        return loss, error.detach()


class LossFnForces(torch.nn.Module):
    def __init__(self, **args):
        super().__init__()
        if args.get('loss_type_forces') is not None:
            args['loss_type'] = args['loss_type_forces']
        if args.get('loss_weight_type_forces') is not None:
            args['loss_weight_type'] = args['loss_weight_type_forces']
        self.error_fn = ErrorFn(**args)

    def forward(self, input, target, batch, weights=None):
        node_weights = weights[batch, None] if weights is not None else None
        error, loss = _weighted_error(self.error_fn, input, target, node_weights)
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

    def forward(self, input, target, weights=None):
        graph_weights = weights[:, None, None] if weights is not None else None
        error, loss = _weighted_error(self.error_fn, input, target, graph_weights)
        return loss, error.detach().mean(dim=(1, 2))


class LossFnRelativeEnergy(torch.nn.Module):
    def __init__(self, target_state: int, **args):
        super().__init__()
        args = dict(args)
        if args.get('loss_type_energy') is not None:
            args['loss_type'] = args['loss_type_energy']
        args['loss_weight_type'] = None
        self.error_fn = ErrorFn(**args)
        self.target_state = int(target_state)

    def pairs(self, reaction_group_id, state_id, valid=None):
        for group_id in torch.unique(reaction_group_id[reaction_group_id >= 0]):
            group_mask = reaction_group_id == group_id
            if valid is not None:
                group_mask = group_mask & valid
            reference_mask = group_mask & (state_id == 0)
            target_mask = group_mask & (state_id == self.target_state)
            if torch.any(reference_mask) and torch.any(target_mask):
                yield reference_mask, target_mask

    def forward(self, input, target, reaction_group_id, state_id, valid=None):
        input = input.reshape(-1)
        target = target.reshape(-1).to(device=input.device, dtype=input.dtype)
        reaction_group_id = reaction_group_id.reshape(-1).to(device=input.device)
        state_id = state_id.reshape(-1).to(device=input.device)

        pred_diffs = []
        target_diffs = []
        for reference_mask, target_mask in self.pairs(
            reaction_group_id, state_id, valid
        ):
            pred_diffs.append(input[target_mask].mean() - input[reference_mask].mean())
            target_diffs.append(
                target[target_mask].mean() - target[reference_mask].mean()
            )

        if not pred_diffs:
            zero = input.sum() * 0.0
            count = torch.tensor(0.0, device=input.device, dtype=input.dtype)
            return zero, zero.detach(), count

        pred = torch.stack(pred_diffs)
        true = torch.stack(target_diffs)
        error = self.error_fn(pred, true)
        count = torch.tensor(
            float(error.numel()), device=input.device, dtype=input.dtype
        )
        return error.mean(), error.detach(), count


class LossFn(torch.nn.Module):
    def __init__(
        self,
        energy_weight: float = 1.0,
        forces_weight: float = 1.0,
        stress_weight: float = 1.0,
        barrier_weight: float = 0.0,
        reaction_energy_weight: float = 0.0,
        loss_energy_per_atom: bool = True,
        **args,
    ):
        super().__init__()
        self.loss_energy = LossFnEnergy(**args)
        self.loss_forces = LossFnForces(**args)
        self.loss_stress = LossFnStress(**args)
        self.loss_barrier = LossFnRelativeEnergy(target_state=1, **args)
        self.loss_reaction_energy = LossFnRelativeEnergy(target_state=2, **args)

        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.stress_weight = stress_weight
        self.barrier_weight = barrier_weight
        self.reaction_energy_weight = reaction_energy_weight
        self.loss_energy_per_atom = loss_energy_per_atom
        self.weights = {
            'energy': energy_weight,
            'forces': forces_weight,
            'stress': stress_weight,
            'barrier': barrier_weight,
            'reaction_energy': reaction_energy_weight,
        }

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

    @staticmethod
    def _graph_attr(y_true, name: str, default: int, *, length: int, device):
        value = getattr(y_true, name, None)
        if value is None:
            return torch.full((length,), default, device=device, dtype=torch.long)
        value = value.to(device=device, dtype=torch.long).reshape(-1)
        if value.numel() < length:
            padded = torch.full((length,), default, device=device, dtype=torch.long)
            padded[: value.numel()] = value
            return padded
        return value[:length]

    @staticmethod
    def graph_weights(data, component):
        weights = getattr(data, f'{component}_weight', None)
        if weights is None:
            return torch.ones_like(data.y.reshape(-1))
        return weights.to(data.y).reshape(-1)

    def reaction_metadata(self, data):
        length = data.y.numel()
        groups = self._graph_attr(
            data, 'reaction_group_id', -1, length=length, device=data.y.device
        )
        if torch.all(groups < 0):
            groups = self._graph_attr(
                data, 'reaction_id', -1, length=length, device=data.y.device
            )
        states = self._graph_attr(
            data, 'state_id', -1, length=length, device=data.y.device
        )
        return groups, states

    def component_counts(self, data):
        """Denominators for each component, without running the model."""
        counts = {key: data.y.new_zeros(()) for key in self.weights}
        if self.energy_weight > 0:
            counts['energy'] = (self.graph_weights(data, 'energy') != 0).sum()
        if self.forces_weight > 0:
            valid = self.graph_weights(data, 'forces') != 0
            counts['forces'] = valid[data.batch].sum() * data.force.shape[-1]
        if self.stress_weight > 0:
            valid = self.graph_weights(data, 'stress') != 0
            counts['stress'] = (
                valid.sum() * data.stress.shape[-1] * data.stress.shape[-2]
            )
        if self.barrier_weight > 0 or self.reaction_energy_weight > 0:
            groups, states = self.reaction_metadata(data)
            valid = self.graph_weights(data, 'energy') != 0
            for key, fn in (
                ('barrier', self.loss_barrier),
                ('reaction_energy', self.loss_reaction_energy),
            ):
                if self.weights[key] > 0:
                    counts[key] = data.y.new_tensor(
                        sum(1 for _ in fn.pairs(groups, states, valid))
                    )
        return counts

    def forward(self, y_pred, y_true):
        loss = Loss(device=y_true.batch.device)
        energy_weights = self.graph_weights(y_true, 'energy')
        counts = self.component_counts(y_true)

        if self.loss_energy_per_atom:
            num_atoms = y_true.ptr[1:] - y_true.ptr[:-1]
            energy_weights = energy_weights / num_atoms

        e_true = y_true.y.reshape(-1)
        f_true = y_true['force']
        s_true = y_true['stress']

        e_pred = y_pred['energy'].reshape(-1)
        f_pred = y_pred['forces']
        s_pred = y_pred['stress']

        loss_e = loss_f = loss_s = None
        loss_b = loss_r = None
        error_e = error_f = error_s = None
        count_b = count_r = None

        if self.energy_weight > 0.0:
            loss_e, error_e = self.loss_energy(e_pred, e_true, energy_weights)
        if self.forces_weight > 0.0:
            loss_f, error_f = self.loss_forces(
                f_pred, f_true, y_true.batch, self.graph_weights(y_true, 'forces')
            )
        if self.stress_weight > 0.0:
            loss_s, error_s = self.loss_stress(
                s_pred, s_true, self.graph_weights(y_true, 'stress')
            )

        graph_count = e_pred.reshape(-1).numel()
        if self.barrier_weight > 0 or self.reaction_energy_weight > 0:
            reaction_group_id, state_id = self.reaction_metadata(y_true)

        total = self.compute_weighted(loss_e, loss_f, loss_s)
        if not isinstance(total, torch.Tensor):
            total = torch.tensor(total, device=e_pred.device, dtype=e_pred.dtype)

        if self.barrier_weight > 0.0:
            loss_b, _, count_b = self.loss_barrier(
                e_pred, e_true, reaction_group_id, state_id, energy_weights != 0
            )
            total = total + self.barrier_weight * loss_b
        if self.reaction_energy_weight > 0.0:
            loss_r, _, count_r = self.loss_reaction_energy(
                e_pred, e_true, reaction_group_id, state_id, energy_weights != 0
            )
            total = total + self.reaction_energy_weight * loss_r

        loss['total'].value += total
        loss['total'].n += graph_count

        if self.energy_weight > 0.0:
            loss['energy'].value = loss_e
            loss['energy'].n += counts['energy']
        if self.forces_weight > 0.0:
            loss['forces'].value += loss_f
            loss['forces'].n += counts['forces']
        if self.stress_weight > 0.0:
            loss['stress'].value += loss_s
            loss['stress'].n += counts['stress']
        if self.barrier_weight > 0.0:
            loss['barrier'].value += loss_b
            loss['barrier'].n += count_b
        if self.reaction_energy_weight > 0.0:
            loss['reaction_energy'].value += loss_r
            loss['reaction_energy'].n += count_r

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
            args_for_type = {
                **args,
                'loss_type': loss_type,
                'loss_type_energy': loss_type,
                'loss_type_forces': loss_type,
                'loss_type_stress': loss_type,
            }
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
    'LossFnRelativeEnergy',
    'LossFn',
    'LossFnCollection',
]
