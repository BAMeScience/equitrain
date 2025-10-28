from __future__ import annotations

import jax.numpy as jnp
import jraph

from equitrain.argparser import ArgumentError


class JaxLossCollection:
    def __init__(self):
        self.total = []

    def append(self, loss_value: float):
        self.total.append(loss_value)

    def mean(self) -> float:
        if not self.total:
            return float('nan')
        return float(jnp.asarray(self.total).mean())


def _build_energy_loss(apply_fn, energy_weight: float):
    if energy_weight <= 0.0:
        raise ArgumentError(
            'The JAX backend currently requires a positive --energy-weight value.'
        )

    def loss_fn(variables, graph: jraph.GraphsTuple):
        mask = jraph.get_graph_padding_mask(graph).astype(jnp.float32)
        outputs = apply_fn(variables, graph)

        pred_energy = jnp.reshape(
            jnp.asarray(outputs['energy'], dtype=mask.dtype), mask.shape
        )

        raw_energy = getattr(graph.globals, 'energy', None)
        if raw_energy is None:
            raise ValueError('Graph globals must contain energy targets for loss.')
        target_energy = jnp.reshape(
            jnp.asarray(raw_energy, dtype=mask.dtype), mask.shape
        )

        raw_weight = getattr(graph.globals, 'weight', None)
        if raw_weight is None:
            weights = jnp.ones(mask.shape, dtype=mask.dtype)
        else:
            weights = jnp.reshape(
                jnp.asarray(raw_weight, dtype=mask.dtype), mask.shape
            )

        diff = pred_energy - target_energy
        sq_error = diff * diff
        weighted = sq_error * weights * mask

        denom = jnp.maximum(jnp.sum(weights * mask), 1.0)
        base_loss = jnp.sum(weighted) / denom
        return jnp.asarray(energy_weight, dtype=mask.dtype) * base_loss

    return loss_fn


def build_loss_fn(apply_fn, energy_weight: float):
    return _build_energy_loss(apply_fn, energy_weight)


def build_eval_loss(apply_fn, energy_weight: float):
    return _build_energy_loss(apply_fn, energy_weight)


__all__ = [
    'JaxLossCollection',
    'build_loss_fn',
    'build_eval_loss',
]
