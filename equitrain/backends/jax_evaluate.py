from __future__ import annotations

import jax
import numpy as np
from mace_jax.data.utils import AtomicNumberTable as JaxAtomicNumberTable

from equitrain.argparser import ArgsFormatter
from equitrain.backends.common import (
    init_logger,
    validate_evaluate_args,
)
from equitrain.backends.jax_utils import load_model_bundle
from equitrain.backends.jax_wrappers import MaceWrapper as JaxMaceWrapper
from equitrain.backends.jax_loss import build_eval_loss, JaxLossCollection
from equitrain.data.backend_jax import atoms_to_graphs, build_loader, make_apply_fn


def _ensure_forces_not_requested(args):
    if getattr(args, 'forces_weight', 0.0) not in (0.0, None):
        raise NotImplementedError(
            'The current JAX backend only supports energy evaluation.'
        )
    if getattr(args, 'stress_weight', 0.0) not in (0.0, None):
        raise NotImplementedError(
            'The current JAX backend only supports energy evaluation.'
        )


def _evaluate_loop(variables, loss_fn, loader):
    if loader is None:
        return None

    eval_step = jax.jit(loss_fn)
    losses = []
    for graph in loader:
        loss = eval_step(variables, graph)
        losses.append(float(jax.device_get(loss)))

    return float(np.mean(losses)) if losses else None


def evaluate(args):
    validate_evaluate_args(args, 'jax')

    _ensure_forces_not_requested(args)

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
    loss_fn = build_eval_loss(apply_fn, args.energy_weight)
    loss_collection = JaxLossCollection()
    for graph in test_loader:
        loss_value = loss_fn(bundle.params, graph)
        loss_collection.append(float(jax.device_get(loss_value)))

    test_loss = loss_collection.mean()

    logger.log(
        1,
        f'Test loss: {test_loss:.6f}'
        if jnp.isfinite(test_loss)
        else 'No test loss computed',
    )
    return test_loss
