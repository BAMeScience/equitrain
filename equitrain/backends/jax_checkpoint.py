from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from flax import core as flax_core
from flax import serialization

from equitrain.backends.jax_utils import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_PARAMS_NAME,
    ModelBundle,
)
from equitrain.logger import FileLogger


def _sanitize_for_json(obj):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return None


def _list_checkpoint_directories(base_path: Path | str, monitor_target: str):
    base = Path(base_path)
    pattern = rf'^.*best_{monitor_target}_epochs@([0-9]+)_e@([0-9]*\.[0-9]+)$'
    regex = re.compile(pattern)

    matching_dirs: list[Path] = []
    matching_vals: list[float] = []
    matching_epochs: list[int] = []

    for candidate in base.glob('**/best_*'):
        if not candidate.is_dir():
            continue
        match = regex.match(candidate.name)
        if match:
            matching_dirs.append(candidate)
            matching_epochs.append(int(match[1]))
            matching_vals.append(float(match[2]))

    return matching_dirs, matching_vals, matching_epochs


def _find_best_checkpoint(base_path: Path | str, monitor_target: str):
    dirs, vals, epochs = _list_checkpoint_directories(base_path, monitor_target)
    if not dirs:
        return None, None
    min_index = vals.index(min(vals))
    return dirs[min_index], epochs[min_index]


def _find_last_checkpoint(base_path: Path | str, monitor_target: str):
    dirs, _, epochs = _list_checkpoint_directories(base_path, monitor_target)
    if not dirs:
        return None, None
    max_index = epochs.index(max(epochs))
    return dirs[max_index], epochs[max_index]


def _resolve_model_path(base_dir: Path):
    for candidate in (DEFAULT_PARAMS_NAME, 'params.msgpack'):
        candidate_path = base_dir / candidate
        if candidate_path.exists():
            return candidate_path
    return None


def _load_params(params_template, params_path: Path):
    if params_path.suffix != '.msgpack':
        raise ValueError(f'Unsupported JAX checkpoint format: {params_path}')
    return serialization.from_bytes(params_template, params_path.read_bytes())


def _load_opt_state(opt_template, opt_path: Path):
    if not opt_path.exists():
        return opt_template
    return serialization.from_bytes(opt_template, opt_path.read_bytes())


def load_model_state(bundle: ModelBundle, state_dict_path: str) -> ModelBundle:
    params_path = Path(state_dict_path)
    params = _load_params(bundle.params, params_path)
    return replace(bundle, params=flax_core.freeze(params))


def load_checkpoint(
    args,
    bundle: ModelBundle,
    opt_state: Any,
    logger: FileLogger | None = None,
):
    load_checkpoint_dir = getattr(args, 'load_checkpoint', None)
    load_checkpoint_model = getattr(args, 'load_checkpoint_model', None)
    load_best_checkpoint = getattr(args, 'load_best_checkpoint', None)
    load_last_checkpoint = getattr(args, 'load_last_checkpoint', None)
    load_best_checkpoint_model = getattr(args, 'load_best_checkpoint_model', None)
    load_last_checkpoint_model = getattr(args, 'load_last_checkpoint_model', None)
    epochs_start = getattr(args, 'epochs_start', 1)

    checkpoint_dir = None
    args_checkpoint = None

    if load_checkpoint_dir is None and load_best_checkpoint:
        checkpoint_dir, epoch = _find_best_checkpoint(args.output_dir, 'val')
        if checkpoint_dir is not None:
            load_checkpoint_dir = checkpoint_dir
        if epoch is not None:
            epochs_start = epoch + 1

    if load_checkpoint_dir is None and load_last_checkpoint:
        checkpoint_dir, epoch = _find_last_checkpoint(args.output_dir, 'val')
        if checkpoint_dir is not None:
            load_checkpoint_dir = checkpoint_dir
        if epoch is not None:
            epochs_start = epoch + 1

    if load_checkpoint_model is None and load_best_checkpoint_model:
        checkpoint_dir, epoch = _find_best_checkpoint(args.output_dir, 'val')
        if checkpoint_dir is not None:
            model_path = _resolve_model_path(checkpoint_dir)
            if model_path is not None:
                load_checkpoint_model = str(model_path)
        if epoch is not None:
            epochs_start = epoch + 1

    if load_checkpoint_model is None and load_last_checkpoint_model:
        checkpoint_dir, epoch = _find_last_checkpoint(args.output_dir, 'val')
        if checkpoint_dir is not None:
            model_path = _resolve_model_path(checkpoint_dir)
            if model_path is not None:
                load_checkpoint_model = str(model_path)
        if epoch is not None:
            epochs_start = epoch + 1

    loaded = False

    if load_checkpoint_dir is not None:
        directory = Path(load_checkpoint_dir)
        if not directory.exists():
            raise FileNotFoundError(f'Checkpoint directory not found: {directory}')
        if logger is not None:
            logger.log(1, f'Loading JAX checkpoint from {directory}')

        params_path = _resolve_model_path(directory)
        if params_path is None:
            raise FileNotFoundError(
                f'Unable to locate parameters inside checkpoint directory: {directory}'
            )

        params = _load_params(bundle.params, params_path)
        bundle = replace(bundle, params=flax_core.freeze(params))

        opt_path = directory / 'opt_state.msgpack'
        if opt_state is not None:
            opt_state = _load_opt_state(opt_state, opt_path)

        args_path = directory / 'args.json'
        if args_path.exists():
            args_checkpoint = json.loads(args_path.read_text())

        loaded = True

    if load_checkpoint_model is not None:
        model_path = Path(load_checkpoint_model)
        if not model_path.exists():
            raise FileNotFoundError(f'Checkpoint model not found: {model_path}')
        if logger is not None:
            logger.log(1, f'Loading JAX model checkpoint {model_path}')
        params = _load_params(bundle.params, model_path)
        bundle = replace(bundle, params=flax_core.freeze(params))
        loaded = True

    if loaded:
        args.epochs_start = epochs_start

    return bundle, opt_state, args_checkpoint


def save_checkpoint(
    args,
    epoch: int,
    val_metric,
    bundle: ModelBundle,
    opt_state: Any,
    logger: FileLogger | None = None,
):
    monitor_total = val_metric.main.meters['total'].avg
    output_dir = Path(args.output_dir) / f'best_val_epochs@{epoch}_e@{monitor_total:0.4g}'
    output_dir.mkdir(parents=True, exist_ok=True)

    if logger is not None:
        logger.log(1, f'Saving JAX checkpoint to `{output_dir}`')

    params_path = output_dir / DEFAULT_PARAMS_NAME
    params_path.write_bytes(serialization.to_bytes(bundle.params))

    opt_path = output_dir / 'opt_state.msgpack'
    if opt_state is not None:
        opt_path.write_bytes(serialization.to_bytes(opt_state))

    config_path = output_dir / DEFAULT_CONFIG_NAME
    config_path.write_text(json.dumps(bundle.config))

    args_path = output_dir / 'args.json'
    args_dict = {k: _sanitize_for_json(v) for k, v in vars(args).items()}
    args_path.write_text(json.dumps(args_dict, default=_sanitize_for_json))


__all__ = ['load_model_state', 'load_checkpoint', 'save_checkpoint']
