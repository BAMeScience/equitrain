from __future__ import annotations

import jax
import jax.numpy as jnp
from mace_jax.data.utils import AtomicNumberTable as JaxAtomicNumberTable

from equitrain.argparser import ArgsFormatter
from equitrain.backends.common import (
    init_logger,
    validate_evaluate_args,
)
from equitrain.backends.jax_utils import load_model_bundle
from equitrain.backends.jax_wrappers import MaceWrapper as JaxMaceWrapper
from equitrain.backends.jax_loss import (
    JaxLossCollection,
    LossSettings,
    build_eval_loss,
)
from equitrain.data.backend_jax import atoms_to_graphs, build_loader, make_apply_fn


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

    def _aux_to_metrics(aux):
        aux_host = jax.device_get(aux)
        return {
            key: (float(value), float(count))
            for key, (value, count) in aux_host['metrics'].items()
        }

    loss_collection = JaxLossCollection()
    for graph in test_loader:
        _, aux = loss_fn(bundle.params, graph)
        loss_collection.update_from_metrics(_aux_to_metrics(aux))

    test_values = loss_collection.as_dict()
    test_total = test_values['total']

    if jnp.isfinite(test_total):
        message = [f'Test loss: total={test_total:.6f}']
        if loss_settings.energy_weight > 0.0:
            message.append(f'energy={test_values["energy"]:.6f}')
        if loss_settings.forces_weight > 0.0:
            message.append(f'forces={test_values["forces"]:.6f}')
        if loss_settings.stress_weight > 0.0:
            message.append(f'stress={test_values["stress"]:.6f}')
        logger.log(1, ', '.join(message))
    else:
        logger.log(1, 'No test loss computed')

    return test_total
