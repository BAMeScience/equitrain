from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jraph
import optax
from flax import serialization, struct
from jax import tree_util as jtu

from equitrain.argparser import (
    ArgsFormatter,
    check_args_consistency,
    validate_training_args,
)
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
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_jax import atoms_to_graphs, build_loader, make_apply_fn
from equitrain.logger import ensure_output_dir, init_logger

from .jax_optimizer import (
    create_optimizer,
    optimizer_kwargs,
)
from .jax_scheduler import create_scheduler_controller

try:  # pragma: no cover - tqdm is optional
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None


ensure_multiprocessing_spawn()


@struct.dataclass
class TrainState:
    params: object
    opt_state: object
    ema_params: object


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
        arr = jnp.asarray(x)
        if 'float0' in str(arr.dtype):
            arr = jnp.zeros_like(arr, dtype=jnp.float32)
        arr = jnp.nan_to_num(arr)
        if clip_value is not None and clip_value > 0.0:
            arr = jnp.clip(arr, -clip_value, clip_value)
        return arr

    return jtu.tree_map(_sanitize, grads)


def _is_multi_device() -> bool:
    return jax.local_device_count() > 1


def _replicate_state(state: TrainState) -> TrainState:
    return jax.device_put_replicated(state, jax.local_devices())


def _unreplicate(tree):
    host = jax.device_get(tree)
    if isinstance(host, list | tuple) and len(host) == jax.local_device_count():
        return jtu.tree_map(lambda x: x[0], host)
    return host


def _prepare_single_batch(graph):
    def _to_device_array(x):
        if x is None:
            return None
        return jnp.asarray(x)

    return jtu.tree_map(_to_device_array, graph, is_leaf=lambda leaf: leaf is None)


def _split_graphs_for_devices(graph, num_devices: int) -> list[list[jraph.GraphsTuple]]:
    graphs = (
        list(jraph.unbatch(graph)) if isinstance(graph, jraph.GraphsTuple) else [graph]
    )
    total = len(graphs)
    if total % num_devices != 0:
        raise ValueError(
            'For JAX multi-device training, batch size must be divisible by the number of devices.'
        )
    per_device = total // num_devices
    return [graphs[i * per_device : (i + 1) * per_device] for i in range(num_devices)]


def _prepare_sharded_batch(graph, num_devices: int):
    chunks = _split_graphs_for_devices(graph, num_devices)
    device_batches = []
    for chunk in chunks:
        graphs_tuple = chunk[0] if len(chunk) == 1 else jraph.batch_np(chunk)
        device_batches.append(_prepare_single_batch(graphs_tuple))

    def _stack_or_none(*values):
        first = values[0]
        if first is None:
            return None
        return jnp.stack(values)

    return jtu.tree_map(
        _stack_or_none, *device_batches, is_leaf=lambda leaf: leaf is None
    )


def _build_train_functions(
    loss_fn,
    optimizer,
    *,
    grad_clip_value,
    use_ema: bool,
    multi_device: bool,
):
    clip_value = None if grad_clip_value is None else float(grad_clip_value)

    if multi_device:

        def grad_step(params, batch):
            (loss, aux), grads = jax.value_and_grad(
                loss_fn, has_aux=True, allow_int=True
            )(params, batch)
            grads = _sanitize_grads(grads, clip_value)
            grads = jax.lax.pmean(grads, axis_name='device')
            loss = jax.lax.pmean(loss, axis_name='device')
            aux = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='device'), aux)
            return loss, aux, grads

        grad_step_fn = jax.pmap(grad_step, axis_name='device', in_axes=(0, 0))

        def apply_updates(state: TrainState, grads, ema_factor):
            updates, new_opt_state = optimizer.update(
                grads, state.opt_state, state.params
            )
            new_params = optax.apply_updates(state.params, updates)
            if use_ema and state.ema_params is not None:
                coeff = jnp.asarray(ema_factor, dtype=jnp.float32)
                new_ema = jtu.tree_map(
                    lambda ema, new: coeff * ema + (1.0 - coeff) * new,
                    state.ema_params,
                    new_params,
                )
            else:
                new_ema = state.ema_params
            return (
                TrainState(
                    params=new_params, opt_state=new_opt_state, ema_params=new_ema
                ),
            )

        apply_updates_fn = jax.pmap(
            apply_updates,
            axis_name='device',
            in_axes=(0, 0, None),
        )
        return grad_step_fn, apply_updates_fn

    def grad_step(params, batch):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)(
            params, batch
        )
        grads = _sanitize_grads(grads, clip_value)
        return loss, aux, grads

    def apply_updates(state: TrainState, grads, ema_factor):
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        if use_ema and state.ema_params is not None:
            coeff = jnp.asarray(ema_factor, dtype=jnp.float32)
            new_ema = jtu.tree_map(
                lambda ema, new: coeff * ema + (1.0 - coeff) * new,
                state.ema_params,
                new_params,
            )
        else:
            new_ema = state.ema_params
        return TrainState(
            params=new_params, opt_state=new_opt_state, ema_params=new_ema
        )

    return jax.jit(grad_step), jax.jit(apply_updates)


def _build_eval_step(loss_fn, *, multi_device: bool):
    if multi_device:

        def step(params, batch):
            loss, aux = loss_fn(params, batch)
            loss = jax.lax.pmean(loss, axis_name='device')
            aux = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='device'), aux)
            return loss, aux

        return jax.pmap(step, axis_name='device', in_axes=(0, 0))

    return jax.jit(loss_fn)


def _run_train_epoch(
    state: TrainState,
    train_loader,
    grad_step_fn,
    apply_updates_fn,
    *,
    max_steps,
    multi_device: bool,
    use_ema: bool,
    ema_decay: float | None,
    ema_count_start: int,
    logger,
    args,
    epoch: int,
    metric_settings,
    learning_rate: float,
    mask,
):
    loss_collection = JaxLossCollection()
    ema_count = ema_count_start
    device_count = jax.local_device_count() if multi_device else 1
    total_steps = None
    if hasattr(train_loader, '__len__'):
        total_steps = len(train_loader)
    if max_steps is not None:
        if total_steps is not None:
            total_steps = min(total_steps, max_steps)
        else:
            total_steps = max_steps

    mask_tree = None
    if mask is not None:
        mask_tree = jtu.tree_map(lambda v: jnp.asarray(v, dtype=jnp.bool_), mask)

    use_tqdm = bool(getattr(args, 'tqdm', False) and tqdm is not None)
    iterator = enumerate(train_loader)
    progress = None
    if use_tqdm:
        progress = tqdm(
            iterator,
            total=total_steps,
            disable=False,
            desc='Training',
        )
        iterator = progress

    for step_index, graph in iterator:
        if max_steps is not None and step_index >= max_steps:
            break

        if isinstance(graph, list):
            micro_batches = [g for g in graph if g is not None]
        else:
            micro_batches = [graph]

        if not micro_batches:
            continue

        micro_count = len(micro_batches)
        inv_micro = 1.0 / float(micro_count)

        step_start = time.perf_counter()

        params_before = state.params
        ema_before = state.ema_params

        ema_factor = 0.0
        if use_ema and ema_decay is not None:
            ema_count += 1
            warmup_decay = (1.0 + ema_count) / (10.0 + ema_count)
            ema_factor = float(min(float(ema_decay), warmup_decay))

        accum_grads = jtu.tree_map(lambda x: jnp.zeros_like(x), state.params)
        macro_collection = JaxLossCollection()

        for micro_batch in micro_batches:
            if multi_device:
                prepared_batch = _prepare_sharded_batch(micro_batch, device_count)
                _, aux_dev, grads = grad_step_fn(state.params, prepared_batch)
                grads = jtu.tree_map(lambda g: g * inv_micro, grads)
                accum_grads = jtu.tree_map(lambda acc, g: acc + g, accum_grads, grads)
                aux_host = _unreplicate(aux_dev)
            else:
                prepared_batch = _prepare_single_batch(micro_batch)
                _, aux_val, grads = grad_step_fn(state.params, prepared_batch)
                grads = jtu.tree_map(lambda g: g * inv_micro, grads)
                accum_grads = jtu.tree_map(lambda acc, g: acc + g, accum_grads, grads)
                aux_host = jax.device_get(aux_val)

            update_collection_from_aux(loss_collection, aux_host)
            update_collection_from_aux(macro_collection, aux_host)

        state = apply_updates_fn(state, accum_grads, ema_factor)

        if mask_tree is not None:
            restored_params = jtu.tree_map(
                lambda new_val, old_val, mask_val: jnp.where(
                    mask_val, new_val, old_val
                ),
                state.params,
                params_before,
                mask_tree,
            )
            if use_ema and state.ema_params is not None and ema_before is not None:
                restored_ema = jtu.tree_map(
                    lambda new_val, old_val, mask_val: jnp.where(
                        mask_val, new_val, old_val
                    ),
                    state.ema_params,
                    ema_before,
                    mask_tree,
                )
            else:
                restored_ema = state.ema_params
            state = TrainState(
                params=restored_params,
                opt_state=state.opt_state,
                ema_params=restored_ema,
            )

        print_freq = getattr(args, 'print_freq', None)
        verbose = getattr(args, 'verbose', 1)
        need_step_metrics = (
            verbose > 1
            and print_freq
            and print_freq > 0
            and (
                (step_index + 1) % print_freq == 0
                or (total_steps is not None and step_index + 1 == total_steps)
            )
        )

        step_metrics = LossMetrics(**metric_settings)
        step_metrics.update(macro_collection)

        if progress is not None:
            progress.update(1)
            if step_metrics.main.meters['total'].count:
                desc_loss = step_metrics.main.meters['total'].avg
                progress.set_description(
                    f'Training (lr={learning_rate:.0e}, loss={desc_loss:.4g})'
                )
            else:
                progress.set_description(f'Training (lr={learning_rate:.0e})')

        if need_step_metrics and step_metrics.main.meters['total'].count:
            step_duration = time.perf_counter() - step_start
            length = total_steps or (
                max_steps if max_steps is not None else step_index + 1
            )
            step_metrics.log_step(
                logger,
                epoch=epoch,
                step=step_index + 1,
                length=length,
                time=step_duration,
                lr=learning_rate,
            )

    if progress is not None:
        progress.close()

    return state, loss_collection, ema_count


def _run_eval_loop(
    params,
    loader,
    eval_step_fn,
    *,
    max_steps,
    multi_device: bool,
):
    if loader is None:
        return None, JaxLossCollection()

    loss_collection = JaxLossCollection()
    device_count = jax.local_device_count() if multi_device else 1
    mean_loss = None

    for step_index, graph in enumerate(loader):
        if max_steps is not None and step_index >= max_steps:
            break
        if isinstance(graph, list):
            micro_batches = [g for g in graph if g is not None]
        else:
            micro_batches = [graph]
        if not micro_batches:
            continue

        for micro_batch in micro_batches:
            if multi_device:
                batch = _prepare_sharded_batch(micro_batch, device_count)
            else:
                batch = _prepare_single_batch(micro_batch)

            loss, aux = eval_step_fn(params, batch)
            loss = _unreplicate(loss) if multi_device else jax.device_get(loss)
            aux = _unreplicate(aux) if multi_device else jax.device_get(aux)
            update_collection_from_aux(loss_collection, aux)
            mean_loss = float(loss)

    if loss_collection.components['total'].count:
        mean_loss = loss_collection.components['total'].value
    else:
        mean_loss = None

    return mean_loss, loss_collection


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

    bundle = load_model_bundle(
        args.model,
        dtype=args.dtype,
        wrapper=getattr(args, 'model_wrapper', None),
    )

    atomic_numbers = bundle.config.get('atomic_numbers')
    if not atomic_numbers:
        raise RuntimeError('Model configuration is missing `atomic_numbers`.')
    z_table = AtomicNumberTable(list(atomic_numbers))

    r_max = float(bundle.config.get('r_max', 0.0))
    if r_max <= 0.0:
        raise RuntimeError('Model configuration must define a positive `r_max`.')

    reduce_cells = getattr(args, 'niggli_reduce', False)
    train_graphs = atoms_to_graphs(
        args.train_file, r_max, z_table, niggli_reduce=reduce_cells
    )
    valid_graphs = atoms_to_graphs(
        args.valid_file, r_max, z_table, niggli_reduce=reduce_cells
    )

    if not train_graphs:
        raise RuntimeError('Training dataset is empty.')

    train_seed = getattr(args, 'seed', None)

    train_loader = build_loader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
        drop=getattr(args, 'batch_drop', False),
        seed=train_seed,
    )
    valid_loader = build_loader(
        valid_graphs,
        batch_size=args.batch_size,
        shuffle=False,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
        drop=getattr(args, 'batch_drop', False),
        seed=train_seed,
    )

    wrapper = JaxMaceWrapper(
        module=bundle.module,
        config=bundle.config,
        compute_force=args.forces_weight > 0.0,
        compute_stress=args.stress_weight > 0.0,
    )

    num_species = len(z_table)
    multi_device = _is_multi_device()
    device_count = jax.local_device_count() if multi_device else 1
    if multi_device and device_count > 0 and (args.batch_size % device_count != 0):
        raise ValueError(
            'For JAX multi-device training, --batch-size must be divisible by the number of local devices.'
        )

    apply_fn = make_apply_fn(wrapper, num_species=num_species)
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

    bundle, opt_state, args_checkpoint = jax_checkpoint.load_checkpoint(
        args, bundle, opt_state, logger
    )
    if args_checkpoint is not None:
        check_args_consistency(args, args_checkpoint, logger)

    use_ema = bool(getattr(args, 'ema', False))
    ema_decay = float(getattr(args, 'ema_decay', 0.999)) if use_ema else None
    ema_params = bundle.params if use_ema else None
    ema_count = 0 if use_ema else 0

    train_state = TrainState(
        params=bundle.params, opt_state=opt_state, ema_params=ema_params
    )
    if multi_device:
        train_state = _replicate_state(train_state)

    grad_clip_value = getattr(args, 'gradient_clipping', None)
    train_max_steps = _normalize_max_steps(getattr(args, 'train_max_steps', None))
    valid_max_steps = _normalize_max_steps(getattr(args, 'valid_max_steps', None))

    grad_step_fn, apply_updates_fn = _build_train_functions(
        loss_fn,
        optimizer,
        grad_clip_value=grad_clip_value,
        use_ema=use_ema,
        multi_device=multi_device,
    )
    eval_step_fn = _build_eval_step(loss_fn, multi_device=multi_device)

    def _host(tree):
        return _unreplicate(tree) if multi_device else tree

    lr_history: list[float] = [current_lr]
    best_val = None
    best_epoch = None
    start_epoch = getattr(args, 'epochs_start', 1)

    metric_settings = dict(
        include_energy=loss_settings.energy_weight > 0.0,
        include_forces=loss_settings.forces_weight > 0.0,
        include_stress=loss_settings.stress_weight > 0.0,
        loss_label=loss_settings.loss_type,
    )

    params_for_eval = (
        train_state.ema_params
        if use_ema and train_state.ema_params is not None
        else train_state.params
    )
    initial_val_loss, initial_val_collection = _run_eval_loop(
        params_for_eval,
        valid_loader,
        eval_step_fn,
        max_steps=valid_max_steps,
        multi_device=multi_device,
    )

    current_params_host = _host(train_state.params)
    current_ema_host = (
        _host(train_state.ema_params)
        if use_ema and train_state.ema_params is not None
        else None
    )
    best_params_host = current_params_host
    best_ema_params_host = current_ema_host

    if valid_loader is not None and initial_val_collection.components['total'].count:
        val_metric = LossMetrics(**metric_settings)
        val_metric.update(initial_val_collection)
        val_metric.log(logger, 'val', epoch=args.epochs_start - 1)
        if wandb_run is not None:
            wandb_run.log(
                {
                    'val_loss': float(val_metric.main.meters['total'].avg),
                    'lr': current_lr,
                    'epoch': args.epochs_start - 1,
                },
                step=args.epochs_start - 1,
            )
        if initial_val_loss is not None and start_epoch > 1:
            best_val = float(initial_val_loss)
            best_epoch = start_epoch - 1
    scheduler_controller.register_initial_metric(
        initial_val_loss, epoch=args.epochs_start - 1
    )

    num_epochs = args.epochs
    last_train_metrics = JaxLossCollection()

    for epoch_offset in range(num_epochs):
        epoch = start_epoch + epoch_offset
        epoch_start_time = time.perf_counter()

        epoch_lr = scheduler_controller.current_lr
        train_state, train_metrics_collection, ema_count = _run_train_epoch(
            train_state,
            train_loader,
            grad_step_fn,
            apply_updates_fn,
            max_steps=train_max_steps,
            multi_device=multi_device,
            use_ema=use_ema,
            ema_decay=ema_decay,
            ema_count_start=ema_count,
            logger=logger,
            args=args,
            epoch=epoch,
            metric_settings=metric_settings,
            learning_rate=epoch_lr,
            mask=mask,
        )
        epoch_time = time.perf_counter() - epoch_start_time
        last_train_metrics = train_metrics_collection

        current_params_host = _host(train_state.params)
        current_ema_host = (
            _host(train_state.ema_params)
            if use_ema and train_state.ema_params is not None
            else None
        )

        eval_params = (
            train_state.ema_params
            if use_ema and train_state.ema_params is not None
            else train_state.params
        )
        val_loss_value, val_metrics_collection = _run_eval_loop(
            eval_params,
            valid_loader,
            eval_step_fn,
            max_steps=valid_max_steps,
            multi_device=multi_device,
        )

        train_metric = LossMetrics(**metric_settings)
        train_metric.update(train_metrics_collection)
        train_metric.log(logger, 'train', epoch=epoch, time=epoch_time, lr=epoch_lr)
        train_loss_value = (
            float(train_metric.main.meters['total'].avg)
            if train_metric.main.meters['total'].count
            else None
        )

        val_metric = LossMetrics(**metric_settings)
        val_metric.update(val_metrics_collection)
        if val_metric.main.meters['total'].count:
            val_metric.log(logger, 'val', epoch=epoch)

        if wandb_run is not None:
            payload = {'epoch': epoch, 'lr': epoch_lr}
            if train_loss_value is not None:
                payload['train_loss'] = train_loss_value
            if val_loss_value is not None:
                payload['val_loss'] = float(val_loss_value)
            wandb_run.log(payload, step=epoch)

        improved = False
        if val_loss_value is None:
            best_val = None
            best_params_host = current_params_host
            best_ema_params_host = current_ema_host
            best_epoch = epoch
            improved = True
        elif best_val is None or val_loss_value < best_val:
            best_val = float(val_loss_value)
            best_params_host = current_params_host
            best_ema_params_host = current_ema_host
            best_epoch = epoch
            improved = True

        if improved:
            opt_state_host = _host(train_state.opt_state)
            jax_checkpoint.save_checkpoint(
                args,
                epoch,
                val_metric,
                ModelBundle(
                    config=bundle.config,
                    params=best_params_host,
                    module=bundle.module,
                ),
                opt_state_host,
                logger,
            )

        monitored_metric = (
            train_loss_value
            if scheduler_controller.monitor == 'train'
            else val_loss_value
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
            grad_step_fn, apply_updates_fn = _build_train_functions(
                loss_fn,
                optimizer,
                grad_clip_value=grad_clip_value,
                use_ema=use_ema,
                multi_device=multi_device,
            )
            logger.log(
                1,
                f'Epoch [{epoch:>4}] -- New learning rate: {current_lr:.4g}',
            )
        else:
            current_lr = scheduler_controller.current_lr

        lr_history.append(current_lr)

    final_params_host = (
        best_ema_params_host
        if use_ema and best_ema_params_host is not None
        else best_params_host
    )

    _save_parameters(Path(args.output_dir), final_params_host)

    test_metrics = None
    if getattr(args, 'test_file', None):
        test_graphs = atoms_to_graphs(
            args.test_file, r_max, z_table, niggli_reduce=reduce_cells
        )
        test_loader = build_loader(
            test_graphs,
            batch_size=args.batch_size,
            shuffle=False,
            max_nodes=args.batch_max_nodes,
            max_edges=args.batch_max_edges,
            seed=train_seed,
        )
        if test_loader is not None:
            eval_params = (
                jax.device_put_replicated(final_params_host, jax.local_devices())
                if multi_device
                else final_params_host
            )
            _, test_metric_collection = _run_eval_loop(
                eval_params,
                test_loader,
                eval_step_fn,
                max_steps=None,
                multi_device=multi_device,
            )
            test_metrics = LossMetrics(
                include_energy=loss_settings.energy_weight > 0.0,
                include_forces=loss_settings.forces_weight > 0.0,
                include_stress=loss_settings.stress_weight > 0.0,
                loss_label=loss_settings.loss_type,
            )
            test_metrics.update(test_metric_collection)
            test_metrics.log(logger, 'test', epoch=start_epoch + num_epochs - 1)

    if best_epoch is None:
        best_epoch = start_epoch + num_epochs - 1

    summary_train_loss = (
        float(last_train_metrics.components['total'].value)
        if last_train_metrics.components['total'].count
        else float('nan')
    )
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
