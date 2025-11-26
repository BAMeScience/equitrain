"""Utility helpers for JAX backends (model loading)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import core as flax_core
from flax import serialization
from jax import tree_util as jtu

from equitrain.argparser import ArgumentError
from equitrain.backends.jax_wrappers import get_wrapper_builder

DEFAULT_CONFIG_NAME = 'config.json'
DEFAULT_PARAMS_NAME = 'params.msgpack'


@dataclass(frozen=True)
class ModelBundle:
    config: dict
    params: dict
    module: object


def set_jax_dtype(dtype: str) -> None:
    dtype = (dtype or 'float32').lower()
    if dtype == 'float64':
        jax.config.update('jax_enable_x64', True)
    elif dtype in {'float32', 'float16'}:
        jax.config.update('jax_enable_x64', False)
    else:
        raise ArgumentError(f'Unsupported dtype for JAX backend: {dtype}')


def resolve_model_paths(model_arg: str) -> tuple[Path, Path]:
    path = Path(model_arg).expanduser().resolve()

    if path.is_dir():
        config_path = path / DEFAULT_CONFIG_NAME
        params_path = path / DEFAULT_PARAMS_NAME
    elif path.suffix == '.json':
        config_path = path
        params_path = path.with_suffix('.msgpack')
    else:
        params_path = path
        config_path = path.with_suffix('.json')

    if not config_path.exists():
        raise FileNotFoundError(
            f'Unable to locate JAX model configuration at {config_path}'
        )
    if not params_path.exists():
        raise FileNotFoundError(
            f'Unable to locate serialized JAX parameters at {params_path}'
        )

    return config_path, params_path


def _discover_wrapper_name(config: dict, explicit: str | None) -> str:
    if explicit:
        return explicit.strip().lower()

    for key in ('model_wrapper', 'wrapper', 'wrapper_name'):
        value = config.get(key)
        if value:
            return str(value).strip().lower()
    return 'mace'


def load_model_bundle(
    model_arg: str,
    dtype: str,
    *,
    wrapper: str | None = None,
) -> ModelBundle:
    config_path, params_path = resolve_model_paths(model_arg)
    config = json.loads(config_path.read_text())

    set_jax_dtype(dtype)

    wrapper_name = _discover_wrapper_name(config, wrapper)
    build_module = get_wrapper_builder(wrapper_name)

    jax_module, template = build_module(config)
    variables = jax_module.init(jax.random.PRNGKey(0), template)
    variables = serialization.from_bytes(variables, params_path.read_bytes())
    variables = flax_core.freeze(variables)

    return ModelBundle(config=config, params=variables, module=jax_module)


def _none_leaf(value):
    return value is None


def replicate_to_local_devices(tree):
    """Broadcast a pytree so the leading axis matches local device count."""
    device_count = jax.local_device_count()
    if device_count <= 1:
        return tree

    def _replicate(leaf):
        if leaf is None:
            return None
        arr = jnp.asarray(leaf)
        broadcast = jnp.broadcast_to(arr, (device_count,) + arr.shape)
        return broadcast

    return jtu.tree_map(_replicate, tree, is_leaf=_none_leaf)


def unreplicate_from_local_devices(tree):
    """Strip a replicated leading axis (if present) from a pytree."""
    device_count = jax.local_device_count()
    if device_count <= 1:
        return tree

    host = jax.device_get(tree)
    if isinstance(host, (list, tuple)) and len(host) == device_count:
        return jtu.tree_map(lambda x: x[0], host, is_leaf=_none_leaf)

    def _maybe_collapse(leaf):
        if leaf is None:
            return None
        arr = np.asarray(leaf)
        if arr.ndim == 0 or arr.shape[0] != device_count:
            return leaf
        first = arr[0]
        if np.all(arr == first):
            return first
        return leaf

    return jtu.tree_map(_maybe_collapse, host, is_leaf=_none_leaf)


__all__ = [
    'ModelBundle',
    'set_jax_dtype',
    'resolve_model_paths',
    'load_model_bundle',
    'replicate_to_local_devices',
    'unreplicate_from_local_devices',
]
