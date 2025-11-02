from __future__ import annotations

from pathlib import Path
import time

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
from .jax_scheduler import create_scheduler_controller


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

    if getattr(args, 'weighted_sampler', False):
        raise ValueError('The JAX backend does not support weighted data sampling.')

    ensure_output_dir(getattr(args, 'output_dir', None))

    logger = init_logger(
        args,
        backend_name='jax',
        enable_logging=True,
        log_to_file=True,
        output_dir=args.output_dir,
    )
    logger.log(1, ArgsFormatter(args))

    wandb_run = None
    if getattr(args, 'wandb_project', None):
        try:
            import wandb
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                'wandb is required for the JAX backend when wandb_project is set.'
            ) from exc

        init_kwargs = {'project': args.wandb_project}
        if getattr(args, 'wandb_name', None):
            init_kwargs['name'] = args.wandb_name
        if getattr(args, 'wandb_group', None):
            init_kwargs['group'] = args.wandb_group
        wandb_run = wandb.init(**init_kwargs, config={'backend': 'jax'})

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
    optimizer_args = optimizer_kwargs(args)
    base_lr = float(optimizer_args.pop('learning_rate'))
    scheduler_controller = create_scheduler_controller(args, base_lr)
    current_lr = scheduler_controller.current_lr
    optimizer = create_optimizer(
        mask=mask,
        learning_rate=current_lr,
        **optimizer_args,
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

    lr_history: list[float] = [current_lr]
    initial_val_loss = None
    best_epoch = None

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

    metric_settings = dict(
        include_energy=loss_settings.energy_weight > 0.0,
        include_forces=loss_settings.forces_weight > 0.0,
        include_stress=loss_settings.stress_weight > 0.0,
        loss_label=loss_settings.loss_type,
    )

    if valid_loader is not None:
        initial_val_loss, initial_val_collection, _ = _evaluate_loop(
            ema_params if use_ema else bundle.params,
            loss_fn,
            valid_loader,
            max_steps=valid_max_steps,
        )
        val_metric = LossMetrics(**metric_settings)
        val_metric.update(initial_val_collection)
        val_metric.log(logger, 'val', epoch=start_epoch - 1)
        if wandb_run is not None and val_metric.main.meters['total'].count:
            wandb_run.log(
                {
                    'val_loss': float(val_metric.main.meters['total'].avg),
                    'lr': current_lr,
                    'epoch': start_epoch - 1,
                },
                step=start_epoch - 1,
            )
        if initial_val_loss is not None:
            best_val = initial_val_loss
            best_params = bundle.params
            best_ema_params = ema_params
            best_epoch = start_epoch - 1
        scheduler_monitor_metric = initial_val_loss
        scheduler_controller.register_initial_metric(
            scheduler_monitor_metric,
            epoch=start_epoch - 1,
        )
    else:
        scheduler_controller.register_initial_metric(None, epoch=start_epoch - 1)

    for epoch_offset in range(num_epochs):
        epoch = start_epoch + epoch_offset
        epoch_start_time = time.perf_counter()

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
        epoch_time = time.perf_counter() - epoch_start_time
        bundle = ModelBundle(
            config=bundle.config, params=updated_params, module=bundle.module
        )

        raw_val_loss, val_metrics, _ = _evaluate_loop(
            ema_params if use_ema else bundle.params,
            loss_fn,
            valid_loader,
            max_steps=valid_max_steps,
        )

        epoch_lr = scheduler_controller.current_lr

        train_metric = LossMetrics(**metric_settings)
        train_metric.update(train_metrics)
        train_metric.log(logger, 'train', epoch=epoch, time=epoch_time, lr=epoch_lr)
        train_loss_value = None
        if train_metric.main.meters['total'].count:
            train_loss_value = float(train_metric.main.meters['total'].avg)

        val_metric = LossMetrics(**metric_settings)
        val_metric.update(val_metrics)
        if val_metric.main.meters['total'].count:
            val_metric.log(logger, 'val', epoch=epoch)
            val_loss_value = float(val_metric.main.meters['total'].avg)
        else:
            val_loss_value = raw_val_loss

        if wandb_run is not None:
            payload = {'epoch': epoch, 'lr': epoch_lr}
            if train_loss_value is not None:
                payload['train_loss'] = train_loss_value
            if val_loss_value is not None:
                payload['val_loss'] = float(val_loss_value)
            wandb_run.log(payload, step=epoch)

        if val_loss_value is None:
            best_params = bundle.params
            best_epoch = epoch
            if use_ema:
                best_ema_params = ema_params
        elif best_val is None or val_loss_value < best_val:
            best_val = float(val_loss_value)
            best_params = bundle.params
            best_epoch = epoch
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

        monitored_metric = (
            train_loss_value if scheduler_controller.monitor == 'train' else val_loss_value
        )
        lr_changed = scheduler_controller.update_after_epoch(
            metric=monitored_metric,
            epoch=epoch,
        )
        if lr_changed:
            current_lr = scheduler_controller.current_lr
            optimizer = create_optimizer(
                mask=mask,
                learning_rate=current_lr,
                **optimizer_args,
            )
            logger.log(
                1,
                f'Epoch [{epoch:>4}] -- New learning rate: {current_lr:.4g}',
            )
        else:
            current_lr = scheduler_controller.current_lr

        lr_history.append(current_lr)

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
            if wandb_run is not None and test_metrics.main.meters['total'].count:
                wandb_run.log(
                    {
                        'test_loss': float(test_metrics.main.meters['total'].avg),
                        'epoch': num_epochs,
                    },
                    step=num_epochs,
                )

    if best_epoch is None:
        best_epoch = start_epoch + num_epochs - 1

    summary_train_loss = float(train_metrics.components['total'].value)
    summary_val_loss = float(best_val) if best_val is not None else None
    summary_test_loss = (
        float(test_metrics.main.meters['total'].avg)
        if test_metrics is not None and test_metrics.main.meters['total'].count
        else None
    )
    summary_initial_val = (
        float(initial_val_loss) if initial_val_loss is not None else None
    )

    if wandb_run is not None:
        wandb_run.finish()

    return {
        'train_loss': summary_train_loss,
        'val_loss': summary_val_loss,
        'test_loss': summary_test_loss,
        'initial_val_loss': summary_initial_val,
        'lr_history': lr_history,
        'best_epoch': best_epoch,
    }


def _save_parameters(output_dir: Path, variables) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = output_dir / 'jax_params.msgpack'
    params_path.write_bytes(serialization.to_bytes(variables))


def evaluate(args):
    from . import jax_evaluate as _jax_evaluate

    return _jax_evaluate.evaluate(args)
