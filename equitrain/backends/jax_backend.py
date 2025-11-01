from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import optax
from flax import serialization
from mace_jax.data.utils import AtomicNumberTable as JaxAtomicNumberTable

from equitrain.argparser import (
    ArgsFormatter,
    check_args_consistency,
    validate_evaluate_args,
    validate_training_args,
)
from equitrain.logger import ensure_output_dir, init_logger
from equitrain.backends import jax_checkpoint
from equitrain.backends.jax_freeze import build_trainable_mask
from equitrain.backends.jax_loss import JaxLossCollection, update_collection_from_aux
from equitrain.backends.jax_loss_fn import LossSettings, build_loss_fn
from equitrain.backends.jax_loss_metrics import LossMetrics
from equitrain.backends.jax_runtime import ensure_multiprocessing_spawn
from equitrain.backends.jax_utils import (
    ModelBundle,
    load_model_bundle,
)
from equitrain.backends.jax_wrappers import MaceWrapper as JaxMaceWrapper
from equitrain.data.backend_jax import atoms_to_graphs, build_loader, make_apply_fn
from .jax_optimizer import (
    create_optimizer,
    optimizer_kwargs,
)
from .jax_scheduler import (
    create_scheduler,
    scheduler_kwargs,
)


ensure_multiprocessing_spawn()


def _normalize_max_steps(value):
    if value is None:
        return None
    try:
        steps = int(value)
    except (TypeError, ValueError):
        return None
    return steps if steps > 0 else None


def _train_loop(
    variables,
    optimizer,
    opt_state,
    train_loader,
    loss_fn,
    *,
    max_steps=None,
):
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def train_step(current_vars, current_opt_state, graph):
        (loss, aux), grads = grad_fn(current_vars, graph)
        updates, new_opt_state = optimizer.update(
            grads, current_opt_state, current_vars
        )
        new_vars = optax.apply_updates(current_vars, updates)
        return new_vars, new_opt_state, loss, aux

    loss_collection = JaxLossCollection()
    per_graph_errors: list[np.ndarray] = []

    for step_index, graph in enumerate(train_loader):
        if max_steps is not None and step_index >= max_steps:
            break
        variables, opt_state, loss, aux = train_step(variables, opt_state, graph)
        per_graph_error = update_collection_from_aux(loss_collection, aux)
        if per_graph_error.size:
            per_graph_errors.append(per_graph_error)

    return variables, opt_state, loss_collection, per_graph_errors


def _evaluate_loop(variables, loss_fn, loader, *, max_steps=None):
    if loader is None:
        return None, JaxLossCollection(), []

    eval_step = jax.jit(loss_fn)
    loss_collection = JaxLossCollection()
    per_graph_errors: list[np.ndarray] = []

    for step_index, graph in enumerate(loader):
        if max_steps is not None and step_index >= max_steps:
            break
        _, aux = eval_step(variables, graph)
        per_graph_error = update_collection_from_aux(loss_collection, aux)
        if per_graph_error.size:
            per_graph_errors.append(per_graph_error)

    mean_loss = loss_collection.components['total'].value
    if not np.isfinite(mean_loss):
        mean_loss = None

    return (mean_loss if loss_collection.components['total'].count else None), loss_collection, per_graph_errors


def train(args):
    validate_training_args(args, 'jax')

    ensure_output_dir(getattr(args, 'output_dir', None))

    logger = init_logger(
        args,
        backend_name='jax',
        enable_logging=True,
        log_to_file=True,
        output_dir=args.output_dir,
    )
    logger.log(1, ArgsFormatter(args))

    bundle = load_model_bundle(args.model, dtype=args.dtype)

    atomic_numbers = bundle.config.get('atomic_numbers')
    if not atomic_numbers:
        raise RuntimeError('Model configuration is missing `atomic_numbers`.')
    z_table = JaxAtomicNumberTable(atomic_numbers)

    r_max = float(bundle.config.get('r_max', 0.0))
    if r_max <= 0.0:
        raise RuntimeError('Model configuration must define a positive `r_max`.')

    train_graphs = atoms_to_graphs(args.train_file, r_max, z_table)
    valid_graphs = atoms_to_graphs(args.valid_file, r_max, z_table)

    if not train_graphs:
        raise RuntimeError('Training dataset is empty.')

    train_seed = getattr(args, 'seed', None)

    train_loader = build_loader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
        seed=train_seed,
    )
    valid_loader = build_loader(
        valid_graphs,
        batch_size=args.batch_size,
        shuffle=False,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
        seed=train_seed,
    )

    wrapper = JaxMaceWrapper(
        module=bundle.module,
        config=bundle.config,
        compute_force=args.forces_weight > 0.0,
        compute_stress=args.stress_weight > 0.0,
    )

    apply_fn = make_apply_fn(wrapper, num_species=len(z_table))
    loss_settings = LossSettings.from_args(args)
    loss_fn = build_loss_fn(apply_fn, loss_settings)
    mask = build_trainable_mask(args, bundle.params, logger)
    schedule = create_scheduler(**scheduler_kwargs(args))
    optimizer = create_optimizer(
        mask=mask,
        learning_rate_schedule=schedule,
        **optimizer_kwargs(args),
    )
    opt_state = optimizer.init(bundle.params)

    bundle, opt_state, args_checkpoint = jax_checkpoint.load_checkpoint(
        args, bundle, opt_state, logger
    )
    if args_checkpoint is not None:
        check_args_consistency(args, args_checkpoint, logger)
    start_epoch = getattr(args, 'epochs_start', 1)

    num_epochs = args.epochs
    start_epoch = args.epochs_start

    train_max_steps = _normalize_max_steps(getattr(args, 'train_max_steps', None))
    valid_max_steps = _normalize_max_steps(getattr(args, 'valid_max_steps', None))

    best_val = None
    best_params = bundle.params
    train_metrics = JaxLossCollection()

    for epoch_offset in range(num_epochs):
        epoch = start_epoch + epoch_offset

        (
            updated_params,
            opt_state,
            train_metrics,
            _,
        ) = _train_loop(
            bundle.params,
            optimizer,
            opt_state,
            train_loader,
            loss_fn,
            max_steps=train_max_steps,
        )
        bundle = ModelBundle(
            config=bundle.config, params=updated_params, module=bundle.module
        )

        val_loss_value, val_metrics, _ = _evaluate_loop(
            bundle.params,
            loss_fn,
            valid_loader,
            max_steps=valid_max_steps,
        )

        train_metric = LossMetrics(
            include_energy=loss_settings.energy_weight > 0.0,
            include_forces=loss_settings.forces_weight > 0.0,
            include_stress=loss_settings.stress_weight > 0.0,
            loss_label=loss_settings.loss_type,
        )
        train_metric.update(train_metrics)

        val_metric = LossMetrics(
            include_energy=loss_settings.energy_weight > 0.0,
            include_forces=loss_settings.forces_weight > 0.0,
            include_stress=loss_settings.stress_weight > 0.0,
            loss_label=loss_settings.loss_type,
        )
        val_metric.update(val_metrics)

        train_metric.log(logger, 'train', epoch=epoch)
        if val_loss_value is not None:
            val_metric.log(logger, 'val', epoch=epoch)

        if val_loss_value is None:
            best_params = bundle.params
        elif best_val is None or val_loss_value < best_val:
            best_val = val_loss_value
            best_params = bundle.params
            jax_checkpoint.save_checkpoint(
                args,
                epoch,
                val_metric,
                bundle,
                opt_state,
                logger,
            )

    _save_parameters(Path(args.output_dir), best_params)

    return {
        'train_loss': train_metrics.components['total'].value,
        'val_loss': best_val,
    }


def _save_parameters(output_dir: Path, variables) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = output_dir / 'jax_params.msgpack'
    params_path.write_bytes(serialization.to_bytes(variables))


def evaluate(args):
    from . import jax_evaluate as _jax_evaluate

    return _jax_evaluate.evaluate(args)
