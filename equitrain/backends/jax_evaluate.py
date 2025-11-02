from __future__ import annotations

import jax
import jax.numpy as jnp
from mace_jax.data.utils import AtomicNumberTable as JaxAtomicNumberTable

from equitrain.argparser import ArgsFormatter, validate_evaluate_args
from equitrain.logger import init_logger
from equitrain.backends.jax_backend import (
    _build_eval_step,
    _is_multi_device,
    _run_eval_loop,
)
from equitrain.backends.jax_loss_fn import LossSettings, build_eval_loss
from equitrain.backends.jax_loss_metrics import LossMetrics
from equitrain.backends.jax_utils import load_model_bundle
from equitrain.backends.jax_wrappers import MaceWrapper as JaxMaceWrapper
from equitrain.backends.jax_runtime import ensure_multiprocessing_spawn
from equitrain.data.backend_jax import atoms_to_graphs, build_loader, make_apply_fn


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

    bundle = load_model_bundle(args.model, dtype=args.dtype)

    atomic_numbers = bundle.config.get('atomic_numbers')
    if not atomic_numbers:
        raise RuntimeError('Model configuration is missing `atomic_numbers`.')
    z_table = JaxAtomicNumberTable(atomic_numbers)

    r_max = float(bundle.config.get('r_max', 0.0))
    if r_max <= 0.0:
        raise RuntimeError('Model configuration must define a positive `r_max`.')

    test_graphs = atoms_to_graphs(args.test_file, r_max, z_table)
    if not test_graphs:
        raise RuntimeError('Test dataset is empty.')

    multi_device = _is_multi_device()
    if multi_device:
        device_count = jax.local_device_count()
        if device_count <= 0:
            raise RuntimeError(
                'JAX reports multi-device mode but no local devices are available.'
            )
        if args.batch_size % device_count != 0:
            raise ValueError(
                'For JAX multi-device evaluation, --batch-size must be divisible by '
                'the number of local devices.'
            )

    test_loader = build_loader(
        test_graphs,
        batch_size=args.batch_size,
        shuffle=False,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
    )

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
        jax.device_put_replicated(bundle.params, jax.local_devices())
        if multi_device
        else bundle.params
    )
    _, loss_collection = _run_eval_loop(
        eval_params,
        test_loader,
        eval_step_fn,
        max_steps=None,
        multi_device=multi_device,
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
