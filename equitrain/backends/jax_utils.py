"""Utility helpers for JAX backends (model loading)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
from flax import serialization
from flax import core as flax_core
from mace_jax.cli import mace_torch2jax

from equitrain.argparser import ArgumentError
from equitrain.backends.jax_runtime import ensure_multiprocessing_spawn


ensure_multiprocessing_spawn()

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


def load_model_bundle(model_arg: str, dtype: str) -> ModelBundle:
    config_path, params_path = resolve_model_paths(model_arg)
    config = json.loads(config_path.read_text())

    set_jax_dtype(dtype)

    jax_module = mace_torch2jax._build_jax_model(config)
    template = mace_torch2jax._prepare_template_data(config)
    variables = jax_module.init(jax.random.PRNGKey(0), template)
    variables = serialization.from_bytes(variables, params_path.read_bytes())
    variables = flax_core.freeze(variables)

    return ModelBundle(config=config, params=variables, module=jax_module)


__all__ = [
    'ModelBundle',
    'set_jax_dtype',
    'resolve_model_paths',
    'load_model_bundle',
]
