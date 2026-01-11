from __future__ import annotations

import itertools
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Suppress PJRT coordination-service shutdown warnings on exit
# (WatchJobStateAsync CANCELLED/UNAVAILABLE); must be set before importing JAX.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('ABSL_MIN_LOG_LEVEL', '2')

import jax
import jax.numpy as jnp
import numpy as np
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
    batched_iterator,
    iter_micro_batches,
    load_model_bundle,
    replicate_to_local_devices,
    set_jax_platform,
    supports_multiprocessing_workers,
    take_chunk,
    unreplicate_from_local_devices,
)
from equitrain.backends.jax_utils import is_multi_device as _is_multi_device
from equitrain.backends.jax_utils import (
    prepare_sharded_batch as _prepare_sharded_batch,
)
from equitrain.backends.jax_utils import (
    prepare_single_batch as _prepare_single_batch,
)
from equitrain.backends.jax_wrappers import MaceWrapper as JaxMaceWrapper
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_jax import get_dataloader, make_apply_fn
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


def _parse_visible_devices() -> list[str] | None:
    raw = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not raw:
        return None
    devices = [device.strip() for device in raw.split(',') if device.strip()]
    return devices or None


def _infer_gpu_count() -> int | None:
    visible_devices = _parse_visible_devices()
    if visible_devices:
        return len(visible_devices)
    slurm_gpus = os.environ.get('SLURM_GPUS_ON_NODE')
    if slurm_gpus:
        try:
            return int(slurm_gpus)
        except ValueError:
            pass
    try:
        return sum(
            1 for dev in jax.devices() if dev.platform in {'cuda', 'gpu', 'rocm'}
        )
    except Exception:
        return None


def _infer_process_count(args) -> int | None:
    if getattr(args, 'process_count', None) is not None:
        return args.process_count
    env_count = os.environ.get('JAX_PROCESS_COUNT')
    if env_count:
        try:
            return int(env_count)
        except ValueError:
            pass
    visible_devices = _parse_visible_devices()
    if visible_devices:
        return len(visible_devices)
    slurm_gpus = os.environ.get('SLURM_GPUS_ON_NODE')
    if slurm_gpus:
        try:
            return int(slurm_gpus)
        except ValueError:
            pass
    try:
        return jax.local_device_count()
    except Exception:
        return None


def _resolve_coordinator_address(args) -> str:
    address = getattr(args, 'coordinator_address', None) or os.environ.get(
        'JAX_COORDINATOR_ADDRESS'
    )
    port = getattr(args, 'coordinator_port', None) or 12345
    if address:
        if ':' in address:
            return address
        return f'{address}:{port}'
    return f'127.0.0.1:{port}'


def _launch_local_processes(args) -> int | None:
    launcher = getattr(args, 'launcher', 'none')
    auto = launcher == 'auto'
    if launcher not in {'local', 'auto'}:
        return None
    if (
        getattr(args, 'process_index', None) is not None
        or os.environ.get('JAX_PROCESS_INDEX') is not None
    ):
        return None
    if auto and not getattr(args, 'distributed', False):
        device = getattr(args, 'jax_platform', None) or getattr(args, 'device', None)
        if device and device.lower() in {'cpu', 'tpu'}:
            return None
        gpu_count = _infer_gpu_count()
        if gpu_count is None or gpu_count <= 1:
            return None
        args.distributed = True
        if getattr(args, 'process_count', None) is None:
            args.process_count = gpu_count

    if not getattr(args, 'distributed', False):
        raise ValueError('Use --distributed together with --launcher local.')

    process_count = _infer_process_count(args)
    if process_count is None:
        raise ValueError(
            'Unable to infer process count. Set --process-count or CUDA_VISIBLE_DEVICES.'
        )
    if process_count < 1:
        raise ValueError(f'Invalid process count: {process_count}.')
    if process_count == 1:
        return None

    visible_devices = _parse_visible_devices()
    if visible_devices and process_count > len(visible_devices):
        raise ValueError(
            'process_count exceeds available CUDA_VISIBLE_DEVICES entries.'
        )
    if visible_devices is None:
        device_count = jax.local_device_count()
        if process_count > device_count:
            raise ValueError(
                f'process_count ({process_count}) exceeds visible devices ({device_count}).'
            )

    coordinator = _resolve_coordinator_address(args)
    base_env = os.environ.copy()
    base_env['JAX_PROCESS_COUNT'] = str(process_count)
    base_env['JAX_COORDINATOR_ADDRESS'] = coordinator
    base_env['EQUITRAIN_LAUNCHED_LOCAL'] = '1'
    base_env.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    base_env.setdefault('ABSL_MIN_LOG_LEVEL', '2')

    child_argv = list(sys.argv)
    if '--distributed' not in child_argv:
        child_argv.append('--distributed')

    procs: list[subprocess.Popen] = []
    previous_handlers = {}

    def _terminate_processes(sig: int, *, force: bool = False) -> None:
        if not procs:
            return
        for proc in procs:
            if proc.poll() is not None:
                continue
            try:
                if proc.pid:
                    os.killpg(proc.pid, sig)
            except Exception:
                try:
                    proc.send_signal(sig)
                except Exception:
                    pass
        if force:
            return
        deadline = time.time() + 10.0
        for proc in procs:
            if proc.poll() is not None:
                continue
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass
        for proc in procs:
            if proc.poll() is not None:
                continue
            try:
                if proc.pid:
                    os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _handle_signal(signum, _frame):
        _terminate_processes(signum)
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _handle_signal)

    try:
        for index in range(process_count):
            env = base_env.copy()
            env['JAX_PROCESS_INDEX'] = str(index)
            if visible_devices:
                env['CUDA_VISIBLE_DEVICES'] = visible_devices[index]
            else:
                env['CUDA_VISIBLE_DEVICES'] = str(index)
            procs.append(
                subprocess.Popen(
                    [sys.executable, *child_argv],
                    env=env,
                    start_new_session=True,
                )
            )

        exit_codes: list[int] = []
        active = list(procs)
        while active:
            for proc in list(active):
                retcode = proc.poll()
                if retcode is None:
                    continue
                exit_codes.append(retcode)
                active.remove(proc)
                if retcode != 0 and active:
                    _terminate_processes(signal.SIGTERM)
                    active.clear()
                    break
            if active:
                time.sleep(0.1)
        _terminate_processes(signal.SIGTERM)
        return max(exit_codes) if exit_codes else 0
    finally:
        _terminate_processes(signal.SIGTERM, force=True)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _initialize_distributed(args) -> None:
    if not getattr(args, 'distributed', False):
        return
    if not hasattr(jax, 'distributed'):
        raise RuntimeError('This JAX build does not expose jax.distributed.initialize.')
    is_initialized = getattr(jax.distributed, 'is_initialized', None)
    if callable(is_initialized) and is_initialized():
        return
    process_count = getattr(args, 'process_count', None) or os.environ.get(
        'JAX_PROCESS_COUNT'
    )
    process_index = getattr(args, 'process_index', None) or os.environ.get(
        'JAX_PROCESS_INDEX'
    )
    coordinator_address = getattr(args, 'coordinator_address', None) or os.environ.get(
        'JAX_COORDINATOR_ADDRESS'
    )
    coordinator = _resolve_coordinator_address(args) if coordinator_address else None
    init_kwargs = {}
    if coordinator is not None:
        init_kwargs['coordinator_address'] = coordinator
    if process_count is not None:
        init_kwargs['num_processes'] = int(process_count)
    if process_index is not None:
        init_kwargs['process_id'] = int(process_index)
    try:
        jax.distributed.initialize(**init_kwargs)
    except RuntimeError as exc:
        if 'already initialized' not in str(exc):
            raise


def _shutdown_distributed() -> None:
    distributed = getattr(jax, 'distributed', None)
    shutdown = getattr(distributed, 'shutdown', None)
    is_initialized = getattr(distributed, 'is_initialized', None)
    if not callable(shutdown):
        return
    try:
        if not callable(is_initialized) or is_initialized():
            process_count = getattr(jax, 'process_count', lambda: 1)()
            if process_count > 1 and os.environ.get('EQUITRAIN_LAUNCHED_LOCAL') == '1':
                return
            shutdown()
    except Exception:
        pass


def _iter_loader_for_epoch(
    loader,
    *,
    epoch: int,
    seed: int | None,
    process_count: int,
    process_index: int,
):
    if loader is None:
        return None
    iter_batches = getattr(loader, 'iter_batches', None)
    if callable(iter_batches):
        return iter_batches(
            epoch=epoch,
            seed=seed,
            process_count=process_count,
            process_index=process_index,
        )
    return loader


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


def _replicate_state(state: TrainState) -> TrainState:
    return replicate_to_local_devices(state)


def _unreplicate(tree):
    return unreplicate_from_local_devices(tree)


def _multi_device_chunk_iterator(loader, device_count: int, *, phase: str, logger):
    """Group per-device micro-batches to feed into ``jax.pmap`` calls."""
    micro_iter = iter_micro_batches(loader)
    first_chunk = take_chunk(micro_iter, device_count)
    if len(first_chunk) < device_count:
        raise RuntimeError(
            f'[{phase}] Need at least {device_count} micro-batches to utilize all '
            'available devices. Increase --batch-max-edges/--batch-max-nodes or '
            'reduce the device count.'
        )

    def _warn(count, expected):
        message = (
            f'[{phase}] Padding incomplete multi-device chunk ({count}/{expected}).'
        )
        if logger is not None:
            logger.log(1, message)
        else:
            print(message)

    def _empty_graph_like(graph):
        nodes = graph.nodes.__class__()
        for key, value in graph.nodes.items():
            nodes[key] = np.zeros_like(value)

        edges = graph.edges.__class__()
        for key, value in graph.edges.items():
            edges[key] = np.zeros_like(value)

        globals_attr = graph.globals
        if globals_attr is None:
            globals_dict = None
        elif hasattr(globals_attr, 'items'):
            globals_dict = globals_attr.__class__()
            for key, value in globals_attr.items():
                globals_dict[key] = np.zeros_like(value)
        else:
            globals_dict = np.zeros_like(globals_attr)

        return graph._replace(
            nodes=nodes,
            edges=edges,
            senders=np.zeros_like(graph.senders),
            receivers=np.zeros_like(graph.receivers),
            globals=globals_dict,
            n_node=np.zeros_like(graph.n_node),
            n_edge=np.zeros_like(graph.n_edge),
        )

    template_graph = next((g for g in first_chunk if g is not None), None)

    def _pad_chunk(chunk):
        filtered = [g for g in chunk if g is not None]
        if len(filtered) >= device_count:
            return filtered
        if not filtered:
            return filtered
        _warn(len(filtered), device_count)
        if template_graph is None:
            raise RuntimeError(
                f'[{phase}] Unable to build padding graphs without a template batch.'
            )
        padded = list(filtered)
        for _ in range(device_count - len(filtered)):
            padded.append(_empty_graph_like(template_graph))
        return padded

    first_chunk = _pad_chunk(first_chunk)

    remainder = batched_iterator(
        micro_iter,
        device_count,
        remainder_action=None,
        drop_remainder=False,
    )
    return itertools.chain(
        [first_chunk], (_pad_chunk(chunk) for chunk in remainder if chunk)
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
        local_devices = jax.local_devices()

        def grad_step(params, batch):
            (loss, aux), grads = jax.value_and_grad(
                loss_fn, has_aux=True, allow_int=True
            )(params, batch)
            grads = _sanitize_grads(grads, clip_value)
            grads = jax.lax.pmean(grads, axis_name='device')
            loss = jax.lax.pmean(loss, axis_name='device')
            aux = jtu.tree_map(lambda x: jax.lax.pmean(x, axis_name='device'), aux)
            return loss, aux, grads

        grad_step_fn = jax.pmap(
            grad_step, axis_name='device', in_axes=(0, 0), devices=local_devices
        )

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
            devices=local_devices,
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
        local_devices = jax.local_devices()

        def step(params, batch):
            loss, aux = loss_fn(params, batch)
            loss = jax.lax.pmean(loss, axis_name='device')
            aux = jtu.tree_map(lambda x: jax.lax.pmean(x, axis_name='device'), aux)
            return loss, aux

        return jax.pmap(step, axis_name='device', in_axes=(0, 0), devices=local_devices)

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
    is_primary: bool,
):
    loss_collection = JaxLossCollection()
    ema_count = ema_count_start
    local_devices = jax.local_devices()
    device_count = len(local_devices) if multi_device else 1
    use_chunked_multi = multi_device and device_count > 1
    total_steps = None
    if hasattr(train_loader, 'total_batches_hint'):
        approx_batches = int(getattr(train_loader, 'total_batches_hint', 0))
    elif hasattr(train_loader, '__len__'):
        approx_batches = len(train_loader)
    else:
        approx_batches = None
    if approx_batches is not None:
        if use_chunked_multi:
            total_steps = (approx_batches + device_count - 1) // device_count
        else:
            total_steps = approx_batches
    if max_steps is not None:
        if total_steps is not None:
            total_steps = min(total_steps, max_steps)
        else:
            total_steps = max_steps

    mask_tree = None
    if mask is not None:
        mask_tree = jtu.tree_map(lambda v: jnp.asarray(v, dtype=jnp.bool_), mask)

    use_tqdm = bool(getattr(args, 'tqdm', False) and tqdm is not None)
    if use_chunked_multi:
        chunk_iter = _multi_device_chunk_iterator(
            train_loader, device_count, phase='Training', logger=logger
        )
        iterator = enumerate(chunk_iter)
    else:
        iterator = enumerate(train_loader)
    progress = None
    if use_tqdm:
        progress = tqdm(
            iterator,
            total=total_steps,
            disable=not is_primary,
            desc='Training',
        )
        iterator = progress

    for step_index, graph in iterator:
        if max_steps is not None and step_index >= max_steps:
            break

        if use_chunked_multi:
            micro_batches = [g for g in graph if g is not None]
            if not micro_batches:
                continue
        else:
            if isinstance(graph, list):
                batch_graph = next((g for g in graph if g is not None), None)
            else:
                batch_graph = graph
            if batch_graph is None:
                continue

        step_start = time.perf_counter()

        params_before = state.params
        ema_before = state.ema_params

        ema_factor = 0.0
        if use_ema and ema_decay is not None:
            ema_count += 1
            warmup_decay = (1.0 + ema_count) / (10.0 + ema_count)
            ema_factor = float(min(float(ema_decay), warmup_decay))

        macro_collection = JaxLossCollection()

        if use_chunked_multi:
            prepared_batch = _prepare_sharded_batch(micro_batches, device_count)
            try:
                _, aux_dev, grads = grad_step_fn(state.params, prepared_batch)
            except jax.errors.JaxRuntimeError as exc:  # pragma: no cover - OOM path
                _raise_memory_hint(exc, args, phase='training')
            accum_grads = grads
            aux_host = _unreplicate(aux_dev)
            update_collection_from_aux(loss_collection, aux_host)
            update_collection_from_aux(macro_collection, aux_host)
        else:
            prepared_batch = _prepare_single_batch(batch_graph)
            try:
                _, aux_val, accum_grads = grad_step_fn(state.params, prepared_batch)
            except jax.errors.JaxRuntimeError as exc:  # pragma: no cover - OOM path
                _raise_memory_hint(exc, args, phase='training')
            aux_host = jax.device_get(aux_val)
            update_collection_from_aux(loss_collection, aux_host)
            update_collection_from_aux(macro_collection, aux_host)

        try:
            state = apply_updates_fn(state, accum_grads, ema_factor)
        except jax.errors.JaxRuntimeError as exc:  # pragma: no cover - OOM path
            _raise_memory_hint(exc, args, phase='training')

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
    logger=None,
):
    if loader is None:
        return None, JaxLossCollection()

    loss_collection = JaxLossCollection()
    local_devices = jax.local_devices()
    device_count = len(local_devices) if multi_device else 1
    mean_loss = None

    if multi_device and device_count > 1:
        data_iter = _multi_device_chunk_iterator(
            loader, device_count, phase='Eval', logger=logger
        )
    else:
        data_iter = loader

    for step_index, graph in enumerate(data_iter):
        if max_steps is not None and step_index >= max_steps:
            break
        if multi_device and device_count > 1:
            micro_batches = graph
            batch = _prepare_sharded_batch(micro_batches, device_count)
            loss, aux = eval_step_fn(params, batch)
            loss = _unreplicate(loss)
            aux = _unreplicate(aux)
            update_collection_from_aux(loss_collection, aux)
            mean_loss = float(loss)
        else:
            if isinstance(graph, list):
                micro_batches = [g for g in graph if g is not None]
            else:
                micro_batches = [graph]
            if not micro_batches:
                continue
            for micro_batch in micro_batches:
                batch = _prepare_single_batch(micro_batch)
                loss, aux = eval_step_fn(params, batch)
                loss = jax.device_get(loss)
                aux = jax.device_get(aux)
                update_collection_from_aux(loss_collection, aux)
                mean_loss = float(loss)

    if loss_collection.components['total'].count:
        mean_loss = loss_collection.components['total'].value
    else:
        mean_loss = None

    return mean_loss, loss_collection


def train(args):
    exit_code = _launch_local_processes(args)
    if exit_code is not None:
        raise SystemExit(exit_code)

    validate_training_args(args, 'jax')
    set_jax_platform(getattr(args, 'jax_platform', None))
    _initialize_distributed(args)

    if getattr(args, 'weighted_sampler', False):
        raise ValueError('The JAX backend does not support weighted data sampling.')

    ensure_output_dir(getattr(args, 'output_dir', None))

    process_count = getattr(jax, 'process_count', lambda: 1)()
    process_index = getattr(jax, 'process_index', lambda: 0)()
    is_primary = process_index == 0
    log_suffix = f'.rank{process_index}' if process_index else ''

    logger = init_logger(
        args,
        backend_name='jax',
        enable_logging=True,
        log_to_file=True,
        output_dir=args.output_dir,
        stream=is_primary,
        log_suffix=log_suffix,
    )
    logger.log(1, ArgsFormatter(args))

    wandb_run = None
    if is_primary and getattr(args, 'wandb_project', None):
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
    reduce_cells = bool(getattr(args, 'niggli_reduce', False))
    train_seed = getattr(args, 'seed', None)

    multi_device = _is_multi_device()
    local_devices = jax.local_devices()
    device_count = len(local_devices) if multi_device else 1

    args.batch_size = None
    if getattr(args, 'batch_max_edges', None) is None:
        raise ValueError(
            'JAX backend requires --batch-max-edges to limit greedy graph packing.'
        )
    args.batch_max_nodes = None

    base_workers = max(int(getattr(args, 'num_workers', 0) or 0), 0)
    if base_workers > 0 and supports_multiprocessing_workers():
        effective_workers = base_workers
        if multi_device and device_count > 1:
            effective_workers *= device_count
    else:
        effective_workers = 0
    prefetch_requested = getattr(args, 'prefetch_batches', None)
    if prefetch_requested is None:
        prefetch_batches = effective_workers
    else:
        prefetch_batches = max(int(prefetch_requested or 0), 0)

    def _build_streaming_loader(path: str | None, shuffle: bool):
        if path in (None, 'None'):
            return None
        return get_dataloader(
            data_file=path,
            atomic_numbers=z_table,
            r_max=r_max,
            shuffle=shuffle,
            max_nodes=None,
            max_edges=args.batch_max_edges,
            drop=getattr(args, 'batch_drop', False),
            seed=train_seed if shuffle else None,
            niggli_reduce=reduce_cells,
            prefetch_batches=prefetch_batches,
            num_workers=effective_workers,
            graph_multiple=device_count if multi_device else 1,
        )

    train_loader = _build_streaming_loader(args.train_file, shuffle=args.shuffle)
    valid_loader = _build_streaming_loader(args.valid_file, shuffle=False)
    if train_loader is None:
        raise RuntimeError('Training dataset is empty.')

    wrapper = JaxMaceWrapper(
        module=bundle.module,
        config=bundle.config,
        compute_force=args.forces_weight > 0.0,
        compute_stress=args.stress_weight > 0.0,
    )

    num_species = len(z_table)

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

    bundle, opt_state, args_checkpoint, resume_ema_params = (
        jax_checkpoint.load_checkpoint(args, bundle, opt_state, logger)
    )
    if args_checkpoint is not None:
        check_args_consistency(args, args_checkpoint, logger)

    use_ema = bool(getattr(args, 'ema', False))
    ema_decay = float(getattr(args, 'ema_decay', 0.999)) if use_ema else None
    ema_params = (
        resume_ema_params
        if use_ema and resume_ema_params is not None
        else (bundle.params if use_ema else None)
    )
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
    initial_valid_loader = _iter_loader_for_epoch(
        valid_loader,
        epoch=0,
        seed=None,
        process_count=process_count,
        process_index=process_index,
    )
    initial_val_loss, initial_val_collection = _run_eval_loop(
        params_for_eval,
        initial_valid_loader,
        eval_step_fn,
        max_steps=valid_max_steps,
        multi_device=multi_device,
        logger=logger,
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
    initial_train_loss = None
    if scheduler_controller.monitor == 'train' and train_loader is not None:

        def _train_metric_step(params, batch):
            loss, aux, _ = grad_step_fn(params, batch)
            return loss, aux

        initial_train_loader = _iter_loader_for_epoch(
            train_loader,
            epoch=start_epoch,
            seed=train_seed if getattr(args, 'shuffle', False) else None,
            process_count=process_count,
            process_index=process_index,
        )
        initial_train_loss, _ = _run_eval_loop(
            train_state.params,
            initial_train_loader,
            _train_metric_step,
            max_steps=train_max_steps,
            multi_device=multi_device,
            logger=logger,
        )
    scheduler_controller.register_initial_metric(
        initial_train_loss
        if scheduler_controller.monitor == 'train'
        else initial_val_loss,
        epoch=args.epochs_start - 1,
    )

    num_epochs = args.epochs
    last_train_metrics = JaxLossCollection()

    for epoch_offset in range(num_epochs):
        epoch = start_epoch + epoch_offset
        epoch_start_time = time.perf_counter()

        epoch_lr = scheduler_controller.current_lr
        epoch_train_loader = _iter_loader_for_epoch(
            train_loader,
            epoch=epoch,
            seed=train_seed if getattr(args, 'shuffle', False) else None,
            process_count=process_count,
            process_index=process_index,
        )
        train_state, train_metrics_collection, ema_count = _run_train_epoch(
            train_state,
            epoch_train_loader,
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
            is_primary=is_primary,
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
        epoch_valid_loader = _iter_loader_for_epoch(
            valid_loader,
            epoch=epoch,
            seed=None,
            process_count=process_count,
            process_index=process_index,
        )
        val_loss_value, val_metrics_collection = _run_eval_loop(
            eval_params,
            epoch_valid_loader,
            eval_step_fn,
            max_steps=valid_max_steps,
            multi_device=multi_device,
            logger=logger,
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

        if improved and is_primary:
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
                ema_params=best_ema_params_host if use_ema else None,
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

    if is_primary:
        _save_parameters(Path(args.output_dir), final_params_host)

    test_metrics = None
    if is_primary and getattr(args, 'test_file', None):
        test_loader = _build_streaming_loader(args.test_file, shuffle=False)
        if test_loader is not None:
            eval_params = (
                replicate_to_local_devices(final_params_host)
                if multi_device
                else final_params_host
            )
            epoch_test_loader = _iter_loader_for_epoch(
                test_loader,
                epoch=start_epoch + num_epochs,
                seed=None,
                process_count=process_count,
                process_index=process_index,
            )
            _, test_metric_collection = _run_eval_loop(
                eval_params,
                epoch_test_loader,
                eval_step_fn,
                max_steps=None,
                multi_device=multi_device,
                logger=logger,
            )
            test_metrics = LossMetrics(
                include_energy=loss_settings.energy_weight > 0.0,
                include_forces=loss_settings.forces_weight > 0.0,
                include_stress=loss_settings.stress_weight > 0.0,
                loss_label=loss_settings.loss_type,
            )
            test_metrics.update(test_metric_collection)
            test_metrics.log(logger, 'test', epoch=start_epoch + num_epochs - 1)

    if process_count > 1:
        try:
            from jax.experimental import multihost_utils

            multihost_utils.sync_global_devices('equitrain_train_complete')
        except Exception as exc:
            if is_primary:
                logger.log(1, f'Failed to sync processes at shutdown: {exc}')

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
    _shutdown_distributed()

    return {
        'train_loss': summary_train_loss,
        'val_loss': summary_val_loss,
        'test_loss': summary_test_loss,
        'initial_val_loss': summary_initial_val,
        'lr_history': lr_history,
        'best_epoch': best_epoch,
    }


def _raise_memory_hint(exc, args, *, phase: str):
    message = str(exc)
    if 'RESOURCE_EXHAUSTED' not in message:
        raise exc
    edges = getattr(args, 'batch_max_edges', None)
    nodes = getattr(args, 'batch_max_nodes', None)
    hint = (
        f'JAX {phase} ran out of device memory. '
        'Try lowering --batch-max-edges/--batch-max-nodes '
        f'(currently {edges or "unset"}/{nodes or "unset"}), '
        'reducing --prefetch-batches, or disabling --multi-gpu.'
    )
    raise RuntimeError(f'{hint}\nOriginal error: {message}') from exc


def _save_parameters(output_dir: Path, variables) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = output_dir / 'jax_params.msgpack'
    params_path.write_bytes(serialization.to_bytes(variables))


def evaluate(args):
    from . import jax_evaluate as _jax_evaluate

    return _jax_evaluate.evaluate(args)
