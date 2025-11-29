from __future__ import annotations

import jax
import jax.numpy as jnp

from equitrain.argparser import ArgsFormatter, validate_evaluate_args
from equitrain.backends.jax_backend import (
    _build_eval_step,
    _run_eval_loop,
)
from equitrain.backends.jax_loss_fn import LossSettings, build_eval_loss
from equitrain.backends.jax_loss_metrics import LossMetrics
from equitrain.backends.jax_runtime import ensure_multiprocessing_spawn
from equitrain.backends.jax_utils import (
    is_multi_device as _is_multi_device,
)
from equitrain.backends.jax_utils import (
    load_model_bundle,
    replicate_to_local_devices,
    supports_multiprocessing_workers,
)
from equitrain.backends.jax_wrappers import MaceWrapper as JaxMaceWrapper
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_jax import get_dataloader, make_apply_fn
from equitrain.logger import init_logger

ensure_multiprocessing_spawn()


def evaluate(args):
    validate_evaluate_args(args, 'jax')

    logger = init_logger(
        args,
        backend_name='jax',
        enable_logging=True,
        log_to_file=False,
        output_dir=None,
    )
    logger.log(1, ArgsFormatter(args))

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

    test_file = args.test_file
    if not (test_file.lower().endswith('.h5') or test_file.lower().endswith('hdf5')):
        raise ValueError(
            'JAX evaluation requires datasets stored in HDF5 format. '
            f'Received: {test_file}'
        )
    multi_device = _is_multi_device()
    device_count = jax.local_device_count() if multi_device else 1
    if getattr(args, 'batch_size', None) is None or args.batch_size <= 0:
        raise ValueError('JAX evaluation requires a positive --batch-size.')
    total_batch_size = int(args.batch_size)
    per_device_batch = total_batch_size
    if multi_device and device_count > 1:
        if total_batch_size % device_count != 0:
            raise ValueError(
                'For JAX multi-device evaluation, --batch-size must be divisible by '
                'the number of local devices.'
            )
        per_device_batch = total_batch_size // device_count

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

    test_loader = get_dataloader(
        data_file=test_file,
        atomic_numbers=z_table,
        r_max=r_max,
        batch_size=per_device_batch,
        shuffle=False,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
        drop=getattr(args, 'batch_drop', False),
        niggli_reduce=getattr(args, 'niggli_reduce', False),
        prefetch_batches=prefetch_batches,
        num_workers=effective_workers,
        graph_multiple=1,
    )

    if test_loader is None:
        raise RuntimeError('Test dataset is empty.')

    wrapper = JaxMaceWrapper(
        module=bundle.module,
        config=bundle.config,
        compute_force=args.forces_weight > 0.0,
        compute_stress=args.stress_weight > 0.0,
    )

    apply_fn = make_apply_fn(wrapper, num_species=len(z_table))
    loss_settings = LossSettings.from_args(args)
    loss_fn = build_eval_loss(apply_fn, loss_settings)
    eval_step_fn = _build_eval_step(loss_fn, multi_device=multi_device)
    eval_params = (
        replicate_to_local_devices(bundle.params) if multi_device else bundle.params
    )
    _, loss_collection = _run_eval_loop(
        eval_params,
        test_loader,
        eval_step_fn,
        max_steps=None,
        multi_device=multi_device,
        logger=logger,
    )

    metric = LossMetrics(
        include_energy=loss_settings.energy_weight > 0.0,
        include_forces=loss_settings.forces_weight > 0.0,
        include_stress=loss_settings.stress_weight > 0.0,
        loss_label=loss_settings.loss_type,
    )
    metric.update(loss_collection)

    total = loss_collection.components['total'].value
    if loss_collection.components['total'].count and jnp.isfinite(total):
        metric.log(logger, 'test')
    else:
        logger.log(1, 'No test loss computed')

    return total
