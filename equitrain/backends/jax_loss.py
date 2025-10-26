from __future__ import annotations

import jax.numpy as jnp
import jraph


class JaxLossCollection:
    def __init__(self):
        self.total = []

    def append(self, loss_value: float):
        self.total.append(loss_value)

    def mean(self) -> float:
        if not self.total:
            return float('nan')
        return float(jnp.asarray(self.total).mean())


def build_eval_loss(apply_fn, energy_weight: float):
    def eval_step(variables, graph: jraph.GraphsTuple):
        mask = jraph.get_graph_padding_mask(graph).astype(jnp.float32)
        outputs = apply_fn(variables, graph)

        pred_energy = jnp.reshape(outputs['energy'], mask.shape)
        target_energy = jnp.reshape(jnp.asarray(graph.globals.energy), mask.shape)
        weights = jnp.reshape(jnp.asarray(graph.globals.weight), mask.shape)

        diff = pred_energy - target_energy
        sq_error = diff * diff
        weighted = sq_error * weights * mask

        denom = jnp.maximum(jnp.sum(weights * mask), 1.0)
        loss_value = jnp.sum(weighted) / denom

        return loss_value

    return eval_step


__all__ = [
    'JaxLossCollection',
    'build_eval_loss',
]
