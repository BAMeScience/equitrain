from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import jraph

from equitrain.argparser import ArgumentError
from equitrain.backends.jax_runtime import ensure_multiprocessing_spawn

ensure_multiprocessing_spawn()


_LOSS_TYPE_ALIASES = {
    'l1': 'mae',
    'l2': 'mse',
    'smooth_l1': 'smooth-l1',
    'smoothl1': 'smooth-l1',
    'smoothmae': 'smooth-l1',
    'smooth-mae': 'smooth-l1',
}
_VALID_LOSS_TYPES = {'mae', 'smooth-l1', 'mse', 'huber'}
_VALID_WEIGHT_TYPES = {None, 'groundstate'}


@dataclass(frozen=True)
class LossSettings:
    energy_weight: float = 1.0
    forces_weight: float = 0.0
    stress_weight: float = 0.0
    loss_type: str = 'huber'
    loss_type_energy: str | None = None
    loss_type_forces: str | None = None
    loss_type_stress: str | None = None
    loss_weight_type: str | None = None
    loss_weight_type_energy: str | None = None
    loss_weight_type_forces: str | None = None
    loss_weight_type_stress: str | None = None
    loss_energy_per_atom: bool = True
    smooth_l1_beta: float = 1.0
    huber_delta: float = 0.01
    loss_clipping: float | None = None

    @classmethod
    def from_args(cls, args):
        def _get(name, default=None):
            return getattr(args, name, default)

        base_loss = _get('loss_type', 'huber')
        return cls(
            energy_weight=float(_get('energy_weight', 0.0) or 0.0),
            forces_weight=float(_get('forces_weight', 0.0) or 0.0),
            stress_weight=float(_get('stress_weight', 0.0) or 0.0),
            loss_type=_canonical_loss_type(base_loss),
            loss_type_energy=_canonical_loss_type(
                _get('loss_type_energy', None) or base_loss
            ),
            loss_type_forces=_canonical_loss_type(
                _get('loss_type_forces', None) or base_loss
            ),
            loss_type_stress=_canonical_loss_type(
                _get('loss_type_stress', None) or base_loss
            ),
            loss_weight_type=_canonical_weight_type(_get('loss_weight_type', None)),
            loss_weight_type_energy=_canonical_weight_type(
                _get('loss_weight_type_energy', None)
            ),
            loss_weight_type_forces=_canonical_weight_type(
                _get('loss_weight_type_forces', None)
            ),
            loss_weight_type_stress=_canonical_weight_type(
                _get('loss_weight_type_stress', None)
            ),
            loss_energy_per_atom=bool(_get('loss_energy_per_atom', True)),
            smooth_l1_beta=float(_get('smooth_l1_beta', 1.0) or 1.0),
            huber_delta=float(_get('huber_delta', 0.01) or 0.01),
            loss_clipping=_get('loss_clipping', None),
        )


def _canonical_loss_type(value: str | None) -> str:
    if value is None:
        return 'huber'
    key = value.replace('_', '-').lower()
    key = _LOSS_TYPE_ALIASES.get(key, key)
    if key not in _VALID_LOSS_TYPES:
        raise ArgumentError(f'Invalid loss type for JAX backend: {value}')
    return key


def _canonical_weight_type(value: str | None) -> str | None:
    if value is None:
        return None
    key = value.lower()
    if key not in _VALID_WEIGHT_TYPES:
        raise ArgumentError(f'Invalid loss weighting type for JAX backend: {value}')
    return key


def _rowwise_norm(array: jnp.ndarray) -> jnp.ndarray:
    if array.ndim > 1:
        return jnp.linalg.norm(array, axis=-1)
    return jnp.abs(array)


def _make_error_fn(
    loss_type: str,
    *,
    smooth_l1_beta: float,
    huber_delta: float,
    loss_clipping: float | None,
    weight_type: str | None,
):
    beta = float(smooth_l1_beta)
    delta = float(huber_delta)
    clip_value = None if loss_clipping is None else float(loss_clipping)

    def _smooth_l1(diff):
        abs_diff = jnp.abs(diff)
        quadratic = 0.5 * (diff**2) / beta
        linear = abs_diff - 0.5 * beta
        return jnp.where(abs_diff < beta, quadratic, linear)

    def _huber(diff):
        abs_diff = jnp.abs(diff)
        quadratic = 0.5 * diff**2
        linear = delta * (abs_diff - 0.5 * delta)
        return jnp.where(abs_diff <= delta, quadratic, linear)

    def error_fn(pred, target):
        diff = pred - target
        if loss_type == 'mae':
            error = jnp.abs(diff)
        elif loss_type == 'smooth-l1':
            if beta < 1e-6:
                error = jnp.abs(diff)
            else:
                error = _smooth_l1(diff)
        elif loss_type == 'mse':
            error = diff**2
        elif loss_type == 'huber':
            error = _huber(diff)
        else:  # pragma: no cover - safeguarded by validation
            raise ArgumentError(f'Unsupported loss type: {loss_type}')

        if clip_value is not None:
            error = jnp.clip(error, a_min=None, a_max=clip_value)

        if weight_type == 'groundstate':
            weights = jnp.exp(-1000.0 * _rowwise_norm(target) ** 2) + 1.0
            if error.ndim > 1:
                error = weights[..., None] * error
            else:
                error = weights * error

        return error

    return error_fn


def _node_padding_mask(graph: jraph.GraphsTuple, mask: jnp.ndarray) -> jnp.ndarray:
    total_nodes = graph.nodes.positions.shape[0]
    return jnp.repeat(mask, graph.n_node, total_repeat_length=total_nodes)


def _node_batch_indices(graph: jraph.GraphsTuple) -> jnp.ndarray:
    batch = jnp.arange(graph.n_node.shape[0], dtype=jnp.int32)
    return jnp.repeat(
        batch, graph.n_node, total_repeat_length=graph.nodes.positions.shape[0]
    )


def _energy_component(outputs, graph, mask, settings, error_fn, dtype):
    if 'energy' not in outputs:
        raise ValueError(
            'Model outputs must include energy predictions for loss computation.'
        )

    mask = jnp.asarray(mask, dtype=dtype)
    mask_sum = jnp.sum(mask)
    mask = jnp.where(mask_sum > 0.0, mask, jnp.ones_like(mask))

    pred = jnp.asarray(outputs['energy'], dtype=dtype)
    if pred.ndim == 0:
        pred = pred[None]
    pred = pred[: mask.shape[0]]

    raw_target = getattr(graph.globals, 'energy', None)
    if raw_target is None:
        raise ValueError(
            'Graph globals must contain energy targets for loss computation.'
        )
    target = jnp.asarray(raw_target, dtype=dtype)
    if target.ndim == 0:
        target = target[None]
    target = target[: mask.shape[0]]

    raw_weights = getattr(graph.globals, 'weight', None)
    if raw_weights is None:
        weights = jnp.ones_like(mask, dtype=dtype)
    else:
        weights = jnp.asarray(raw_weights, dtype=dtype)
        if weights.ndim == 0:
            weights = weights[None]
        weights = weights[: mask.shape[0]]

    if settings.loss_energy_per_atom:
        num_atoms = jnp.asarray(graph.n_node, dtype=dtype)
        if num_atoms.ndim == 0:
            num_atoms = num_atoms[None]
        num_atoms = num_atoms[: mask.shape[0]]
        num_atoms = jnp.maximum(num_atoms, 1.0)
        weights = weights / num_atoms

    weights = weights * mask

    error = error_fn(pred, target)
    error = jnp.nan_to_num(error)
    weighted_error = error * weights
    weighted_error = jnp.nan_to_num(weighted_error)

    denom = jnp.maximum(jnp.sum(mask), 1.0)
    loss = jnp.sum(weighted_error) / denom

    count = jnp.sum(mask)
    per_graph_error = weighted_error

    return loss, count, per_graph_error


def _forces_component(outputs, graph, mask, error_fn, dtype):
    if 'forces' not in outputs:
        raise ValueError(
            'Model outputs must include forces when --forces-weight is positive.'
        )

    pred = jnp.asarray(outputs['forces'], dtype=dtype)
    target = jnp.asarray(getattr(graph.nodes, 'forces'), dtype=dtype)

    node_mask = _node_padding_mask(graph, mask).astype(dtype)
    node_mask_sum = jnp.sum(node_mask)
    node_mask = jnp.where(node_mask_sum > 0.0, node_mask, jnp.ones_like(node_mask))
    node_batch = _node_batch_indices(graph)

    error = error_fn(pred, target)
    masked_error = error * node_mask[:, None]

    denom = jnp.maximum(jnp.sum(node_mask) * error.shape[-1], 1.0)
    loss = jnp.sum(masked_error) / denom

    per_node_error = jnp.where(
        node_mask > 0.0,
        jnp.mean(error, axis=-1),
        jnp.zeros_like(node_mask),
    )

    sum_per_graph = jraph.segment_sum(
        per_node_error * node_mask, node_batch, mask.shape[0]
    )
    count_per_graph = jraph.segment_sum(node_mask, node_batch, mask.shape[0])
    per_graph_error = jnp.where(
        count_per_graph > 0,
        sum_per_graph / jnp.maximum(count_per_graph, 1.0),
        jnp.zeros_like(sum_per_graph),
    )
    per_graph_error = per_graph_error * mask

    count = jnp.sum(node_mask) * error.shape[-1]

    return loss, count, per_graph_error


def _stress_component(outputs, graph, mask, error_fn, dtype):
    if 'stress' not in outputs:
        raise ValueError(
            'Model outputs must include stress when --stress-weight is positive.'
        )

    pred = jnp.asarray(outputs['stress'], dtype=dtype)
    raw_target = getattr(graph.globals, 'stress', None)
    if raw_target is None:
        raise ValueError(
            'Graph globals must contain stress targets for loss computation.'
        )
    target = jnp.asarray(raw_target, dtype=dtype)

    if pred.shape != target.shape:
        raise ValueError(
            f'Predicted stress shape {pred.shape} does not match target shape {target.shape}.'
        )

    mask_expanded = mask[:, None, None]
    mask_sum = jnp.sum(mask_expanded)
    mask_expanded = jnp.where(
        mask_sum > 0.0, mask_expanded, jnp.ones_like(mask_expanded)
    )
    mask = jnp.where(mask_sum > 0.0, mask, jnp.ones_like(mask))

    error = error_fn(pred, target)
    masked_error = error * mask_expanded

    denom = jnp.maximum(jnp.sum(mask) * error.shape[-1] * error.shape[-2], 1.0)
    loss = jnp.sum(masked_error) / denom

    per_graph_error = jnp.mean(error, axis=(-2, -1)) * mask

    count = jnp.sum(mask) * error.shape[-1] * error.shape[-2]

    return loss, count, per_graph_error


def build_loss_fn(apply_fn, settings: LossSettings):
    if (
        settings.energy_weight <= 0.0
        and settings.forces_weight <= 0.0
        and settings.stress_weight <= 0.0
    ):
        raise ArgumentError(
            'At least one of the energy, forces, or stress weights must be positive.'
        )

    energy_error_fn = _make_error_fn(
        settings.loss_type_energy,
        smooth_l1_beta=settings.smooth_l1_beta,
        huber_delta=settings.huber_delta,
        loss_clipping=settings.loss_clipping,
        weight_type=settings.loss_weight_type_energy or settings.loss_weight_type,
    )
    forces_error_fn = _make_error_fn(
        settings.loss_type_forces,
        smooth_l1_beta=settings.smooth_l1_beta,
        huber_delta=settings.huber_delta,
        loss_clipping=settings.loss_clipping,
        weight_type=settings.loss_weight_type_forces or settings.loss_weight_type,
    )
    stress_error_fn = _make_error_fn(
        settings.loss_type_stress,
        smooth_l1_beta=settings.smooth_l1_beta,
        huber_delta=settings.huber_delta,
        loss_clipping=settings.loss_clipping,
        weight_type=settings.loss_weight_type_stress or settings.loss_weight_type,
    )

    def loss_fn(variables, graph: jraph.GraphsTuple):
        mask = jraph.get_graph_padding_mask(graph).astype(jnp.float32)
        mask_sum = jnp.sum(mask)
        mask = jnp.where(mask_sum > 0.0, mask, jnp.ones_like(mask))
        outputs = apply_fn(variables, graph)

        dtype = jnp.asarray(graph.nodes.positions).dtype

        if isinstance(outputs, dict):
            outputs = dict(outputs)
            if 'energy' in outputs:
                energy = jnp.asarray(outputs['energy'], dtype=dtype)
                if energy.ndim == 0:
                    energy = energy[None]
                energy_mask = mask[: energy.shape[0]]
                energy = jnp.nan_to_num(energy)
                energy = energy * energy_mask
                outputs['energy'] = energy
            if 'forces' in outputs:
                forces_value = outputs['forces']
                if forces_value is not None:
                    forces = jnp.asarray(forces_value, dtype=dtype)
                    forces = jnp.nan_to_num(forces)
                    node_mask = _node_padding_mask(graph, mask).astype(dtype)
                    forces = forces * node_mask[:, None]
                    outputs['forces'] = forces
            if 'stress' in outputs:
                stress_value = outputs['stress']
                if stress_value is not None:
                    stress = jnp.asarray(stress_value, dtype=dtype)
                    if stress.ndim == 2:
                        stress = stress[None, ...]
                    stress_mask = mask[: stress.shape[0]]
                    stress = jnp.nan_to_num(stress)
                    stress = stress * stress_mask[:, None, None]
                    outputs['stress'] = stress
            for key, value in list(outputs.items()):
                if isinstance(value, jnp.ndarray) and key not in {
                    'energy',
                    'forces',
                    'stress',
                }:
                    outputs[key] = jnp.nan_to_num(value)

        metrics = {
            'total': (jnp.array(0.0, dtype), jnp.array(0.0, dtype)),
            'energy': (jnp.array(0.0, dtype), jnp.array(0.0, dtype)),
            'forces': (jnp.array(0.0, dtype), jnp.array(0.0, dtype)),
            'stress': (jnp.array(0.0, dtype), jnp.array(0.0, dtype)),
        }
        per_graph_error = jnp.zeros(mask.shape, dtype=dtype)
        total_loss = jnp.array(0.0, dtype)

        if settings.energy_weight > 0.0:
            energy_loss, energy_count, energy_error = _energy_component(
                outputs,
                graph,
                mask,
                settings,
                energy_error_fn,
                dtype,
            )
            total_loss = total_loss + settings.energy_weight * energy_loss
            metrics['energy'] = (energy_loss, energy_count)
            per_graph_error = per_graph_error + settings.energy_weight * energy_error

        if settings.forces_weight > 0.0:
            forces_loss, forces_count, forces_error = _forces_component(
                outputs,
                graph,
                mask,
                forces_error_fn,
                dtype,
            )
            total_loss = total_loss + settings.forces_weight * forces_loss
            metrics['forces'] = (forces_loss, forces_count)
            per_graph_error = per_graph_error + settings.forces_weight * forces_error

        if settings.stress_weight > 0.0:
            stress_loss, stress_count, stress_error = _stress_component(
                outputs,
                graph,
                mask,
                stress_error_fn,
                dtype,
            )
            total_loss = total_loss + settings.stress_weight * stress_loss
            metrics['stress'] = (stress_loss, stress_count)
            per_graph_error = per_graph_error + settings.stress_weight * stress_error

        total_count = jnp.sum(mask)
        metrics['total'] = (total_loss, total_count)

        aux = {
            'metrics': metrics,
            'per_graph_error': per_graph_error,
        }
        return total_loss, aux

    return loss_fn


def build_eval_loss(apply_fn, settings: LossSettings):
    loss_fn = build_loss_fn(apply_fn, settings)

    def eval_fn(variables, graph: jraph.GraphsTuple):
        loss_value, aux = loss_fn(variables, graph)
        return loss_value, aux

    return eval_fn


__all__ = ['LossSettings', 'build_loss_fn', 'build_eval_loss']
