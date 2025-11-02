from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from jax import tree_util as jtu
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


def _sanitize_grads(grads, clip_value: float | None):
    def _sanitize(x):
        x = jnp.nan_to_num(x)
        if clip_value is not None and clip_value > 0.0:
            x = jnp.clip(x, -clip_value, clip_value)
        return x

    return jtu.tree_map(_sanitize, grads)


def _train_loop(
    variables,
    optimizer,
    opt_state,
    train_loader,
    loss_fn,
    *,
    max_steps=None,
    grad_clip_value: float | None = None,
    ema_params=None,
    ema_decay: float | None = None,
    ema_count: int | None = None,
):
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    clip_value = None if grad_clip_value is None else float(grad_clip_value)

    @jax.jit
    def train_step(current_vars, current_opt_state, graph):
        (loss, aux), grads = grad_fn(current_vars, graph)
        grads = _sanitize_grads(grads, clip_value)
        updates, new_opt_state = optimizer.update(
            grads, current_opt_state, current_vars
        )
        new_vars = optax.apply_updates(current_vars, updates)
        return new_vars, new_opt_state, loss, aux

    loss_collection = JaxLossCollection()
    per_graph_errors: list[np.ndarray] = []
    ema_active = ema_params is not None and ema_decay is not None
    ema_step_count = ema_count if ema_count is not None else 0

    for step_index, graph in enumerate(train_loader):
        if max_steps is not None and step_index >= max_steps:
            break
        variables, opt_state, loss, aux = train_step(variables, opt_state, graph)
        if ema_active:
            ema_step_count += 1
            warmup_decay = (1.0 + ema_step_count) / (10.0 + ema_step_count)
            decay = min(float(ema_decay), warmup_decay)
            coeff_new = 1.0 - decay
            ema_params = jtu.tree_map(
                lambda ema, new: decay * ema + coeff_new * new,
                ema_params,
                variables,
            )
        per_graph_error = update_collection_from_aux(loss_collection, aux)
        if per_graph_error.size:
            per_graph_errors.append(per_graph_error)

    if not ema_active:
        ema_step_count = ema_count

    return (
        variables,
        opt_state,
        loss_collection,
        per_graph_errors,
        ema_params,
        ema_step_count,
    )


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

    ema_params = None
    ema_decay = None
    ema_count = None
    use_ema = getattr(args, 'ema', False)
    if use_ema:
        ema_params = bundle.params
        ema_decay = float(getattr(args, 'ema_decay', 0.999))
        ema_count = 0

    train_max_steps = _normalize_max_steps(getattr(args, 'train_max_steps', None))
    valid_max_steps = _normalize_max_steps(getattr(args, 'valid_max_steps', None))
    grad_clip_value = getattr(args, 'gradient_clipping', None)

    bundle, opt_state, args_checkpoint = jax_checkpoint.load_checkpoint(
        args, bundle, opt_state, logger
    )
    if args_checkpoint is not None:
        check_args_consistency(args, args_checkpoint, logger)
    start_epoch = getattr(args, 'epochs_start', 1)

    num_epochs = args.epochs
    start_epoch = args.epochs_start

    best_val = None
    best_params = bundle.params
    best_ema_params = ema_params
    train_metrics = JaxLossCollection()

    for epoch_offset in range(num_epochs):
        epoch = start_epoch + epoch_offset

        (
            updated_params,
            opt_state,
            train_metrics,
            _,
            ema_params,
            ema_count,
        ) = _train_loop(
            bundle.params,
            optimizer,
            opt_state,
            train_loader,
            loss_fn,
            max_steps=train_max_steps,
            grad_clip_value=grad_clip_value,
            ema_params=ema_params if use_ema else None,
            ema_decay=ema_decay if use_ema else None,
            ema_count=ema_count if use_ema else None,
        )
        bundle = ModelBundle(
            config=bundle.config, params=updated_params, module=bundle.module
        )

        val_loss_value, val_metrics, _ = _evaluate_loop(
            ema_params if use_ema else bundle.params,
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
            if use_ema:
                best_ema_params = ema_params
        elif best_val is None or val_loss_value < best_val:
            best_val = val_loss_value
            best_params = bundle.params
            if use_ema:
                best_ema_params = ema_params
            jax_checkpoint.save_checkpoint(
                args,
                epoch,
                val_metric,
                ModelBundle(
                    config=bundle.config,
                    params=best_params,
                    module=bundle.module,
                ),
                opt_state,
                logger,
            )

    final_params = best_params
    if use_ema and best_ema_params is not None:
        final_params = best_ema_params

    _save_parameters(Path(args.output_dir), final_params)

    test_metrics = None
    if getattr(args, 'test_file', None):
        test_graphs = atoms_to_graphs(args.test_file, r_max, z_table)
        test_loader = build_loader(
            test_graphs,
            batch_size=args.batch_size,
            shuffle=False,
            max_nodes=args.batch_max_nodes,
            max_edges=args.batch_max_edges,
            seed=train_seed,
        )
        if test_loader is not None:
            _, test_metric_collection, _ = _evaluate_loop(
                final_params,
                loss_fn,
                test_loader,
            )
            test_metrics = LossMetrics(
                include_energy=loss_settings.energy_weight > 0.0,
                include_forces=loss_settings.forces_weight > 0.0,
                include_stress=loss_settings.stress_weight > 0.0,
                loss_label=loss_settings.loss_type,
            )
            test_metrics.update(test_metric_collection)
            test_metrics.log(logger, 'test', epoch=num_epochs)

    return {
        'train_loss': train_metrics.components['total'].value,
        'val_loss': best_val,
        'test_loss': (
            test_metrics.main.meters['total'].avg if test_metrics is not None else None
        ),
    }


def _save_parameters(output_dir: Path, variables) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = output_dir / 'jax_params.msgpack'
    params_path.write_bytes(serialization.to_bytes(variables))


def evaluate(args):
    from . import jax_evaluate as _jax_evaluate

    return _jax_evaluate.evaluate(args)
