from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import core as flax_core
from flax import nnx, traverse_util
from mace_jax.nnx_utils import state_to_pure_dict

_DELTA_PARAM_SUFFIXES = frozenset({'weight', 'bias'})


def _as_mutable_tree(tree):
    return flax_core.unfreeze(tree)


def _is_delta_param_path(path: tuple[object, ...]) -> bool:
    return bool(path) and str(path[-1]) in _DELTA_PARAM_SUFFIXES


def build_delta_template_from_param_tree(param_tree) -> dict:
    flat_params = traverse_util.flatten_dict(_as_mutable_tree(param_tree))
    delta_flat = {
        path: jnp.zeros_like(jnp.asarray(value))
        for path, value in flat_params.items()
        if _is_delta_param_path(path)
    }
    return traverse_util.unflatten_dict(delta_flat)


def build_delta_template(module) -> dict:
    param_state = nnx.state(module, nnx.Param)
    return build_delta_template_from_param_tree(state_to_pure_dict(param_state))


def merge_delta_params(base_params, delta_params) -> dict:
    flat_base = traverse_util.flatten_dict(_as_mutable_tree(base_params))
    flat_delta = traverse_util.flatten_dict(_as_mutable_tree(delta_params))

    for path, delta_value in flat_delta.items():
        flat_base[path] = (
            jax.lax.stop_gradient(jnp.asarray(flat_base[path])) + delta_value
        )

    return traverse_util.unflatten_dict(flat_base)


def ensure_delta_params(variables, delta_template) -> flax_core.FrozenDict:
    """
    Wrap a full MACE-JAX NNX state into Equitrain's delta fine-tuning layout.
    """
    unfrozen = _as_mutable_tree(variables)

    if (
        'base_params' in unfrozen
        and unfrozen.get('params', {}).get('delta') is not None
    ):
        return flax_core.freeze(unfrozen)

    return flax_core.freeze(
        {
            'base_params': unfrozen,
            'params': {'delta': _as_mutable_tree(delta_template)},
        }
    )


class DeltaFineTuneModule:
    """
    Wrap an NNX module so Equitrain can fine-tune additive deltas on top of the
    frozen imported MACE-JAX state.
    """

    def __init__(self, inner_module):
        self._inner = inner_module
        self._graphdef = nnx.graphdef(inner_module)
        self.delta_template = build_delta_template(inner_module)

    def init(self, *_args, **_kwargs):
        _, state = nnx.split(self._inner)
        base_vars = state_to_pure_dict(state)
        return ensure_delta_params(base_vars, self.delta_template)

    def apply(self, variables, *args, **kwargs):
        base_tree = variables['base_params']
        delta_tree = variables['params']['delta']
        actual_vars = merge_delta_params(base_tree, delta_tree)
        outputs, _ = self._graphdef.apply(actual_vars)(*args, **kwargs)
        return outputs


def wrap_with_deltas(module):
    return DeltaFineTuneModule(module)


__all__ = [
    'DeltaFineTuneModule',
    'build_delta_template',
    'build_delta_template_from_param_tree',
    'ensure_delta_params',
    'merge_delta_params',
    'wrap_with_deltas',
]
