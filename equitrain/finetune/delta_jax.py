from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import core as flax_core
from jax import tree_util as jtu


class DeltaFineTuneModule:
    """
    Wrap a Flax module so that parameters are exposed as the sum of frozen ``base``
    weights and trainable ``delta`` offsets.
    """

    def __init__(self, inner_module):
        self._inner = inner_module

    def init(self, rng, *args, **kwargs):
        base_vars = self._inner.init(rng, *args, **kwargs)
        return ensure_delta_params(base_vars)

    def apply(self, variables, *args, **kwargs):
        params = variables.get('params', {})
        base_tree = variables.get('base_params', {})
        if base_tree and 'delta' in params:
            base_tree = jtu.tree_map(jax.lax.stop_gradient, base_tree)
            combined = jtu.tree_map(lambda b, d: b + d, base_tree, params['delta'])
            merged = {
                key: value
                for key, value in variables.items()
                if key not in ('params', 'base_params')
            }
            merged['params'] = combined
            actual_vars = flax_core.freeze(merged)
            return self._inner.apply(actual_vars, *args, **kwargs)
        return self._inner.apply(variables, *args, **kwargs)


def ensure_delta_params(variables: flax_core.FrozenDict) -> flax_core.FrozenDict:
    """
    Normalise a parameter tree so that trainable deltas live under ``params['delta']``
    and the frozen parameters under ``base_params``.
    """
    unfrozen = flax_core.unfreeze(variables)
    params_tree = unfrozen.get('params')

    if params_tree is None:
        return flax_core.freeze(unfrozen)

    if 'base' in params_tree and 'delta' in params_tree:
        base_tree = params_tree['base']
        delta_tree = params_tree['delta']
        unfrozen['params'] = {'delta': delta_tree}
        unfrozen['base_params'] = base_tree
        return flax_core.freeze(unfrozen)

    if 'delta' in params_tree:
        if 'base_params' not in unfrozen:
            base_shape = jtu.tree_map(lambda x: jnp.zeros_like(x), params_tree['delta'])
            unfrozen['base_params'] = base_shape
        return flax_core.freeze(unfrozen)

    base_tree = params_tree
    delta_tree = jtu.tree_map(lambda x: jnp.zeros_like(x), base_tree)
    unfrozen['params'] = {'delta': delta_tree}
    unfrozen['base_params'] = base_tree
    return flax_core.freeze(unfrozen)


def wrap_with_deltas(module):
    """Small convenience helper mirroring the test-suite behaviour."""
    return DeltaFineTuneModule(module)


__all__ = ['DeltaFineTuneModule', 'ensure_delta_params', 'wrap_with_deltas']
