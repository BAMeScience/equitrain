from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from mace_jax.data.utils import AtomicNumberTable as JaxAtomicNumberTable

from equitrain.argparser import ArgsFormatter
from equitrain.backends.common import (
    ensure_output_dir,
    init_logger,
    validate_evaluate_args,
    validate_training_args,
)
from equitrain.backends.jax_freeze import build_trainable_mask
from equitrain.backends.jax_utils import (
    ModelBundle,
    build_loss_fn,
    load_model_bundle,
)
from equitrain.backends.jax_wrappers import MaceWrapper as JaxMaceWrapper
from equitrain.data.backend_jax import atoms_to_graphs, build_loader, make_apply_fn


def _ensure_forces_not_requested(args):
    if getattr(args, 'forces_weight', 0.0) not in (0.0, None):
        raise NotImplementedError(
            'The current JAX backend only supports energy training.'
        )
    if getattr(args, 'stress_weight', 0.0) not in (0.0, None):
        raise NotImplementedError(
            'The current JAX backend only supports energy training.'
        )


def _train_loop(variables, optimizer, opt_state, train_loader, loss_fn):
    grad_fn = jax.value_and_grad(loss_fn)

    @jax.jit
    def train_step(current_vars, current_opt_state, graph):
        loss, grads = grad_fn(current_vars, graph)
        updates, new_opt_state = optimizer.update(
            grads, current_opt_state, current_vars
        )
        new_vars = optax.apply_updates(current_vars, updates)
        return new_vars, new_opt_state, loss

    losses = []
    for graph in train_loader:
        variables, opt_state, loss = train_step(variables, opt_state, graph)
        losses.append(float(jax.device_get(loss)))

    return variables, opt_state, float(np.mean(losses)) if losses else 0.0


def _evaluate_loop(variables, loss_fn, loader):
    if loader is None:
        return None

    eval_step = jax.jit(loss_fn)
    losses = []
    for graph in loader:
        loss = eval_step(variables, graph)
        losses.append(float(jax.device_get(loss)))

    return float(np.mean(losses)) if losses else None


def train(args):
    validate_training_args(args, 'jax')

    _ensure_forces_not_requested(args)

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

    train_loader = build_loader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
    )
    valid_loader = build_loader(
        valid_graphs,
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
    loss_fn = build_loss_fn(apply_fn, args.energy_weight)
    mask = build_trainable_mask(args, bundle.params, logger)
    optimizer = optax.adam(args.lr)
    if mask is not None:
        optimizer = optax.masked(optimizer, mask)
    opt_state = optimizer.init(bundle.params)

    num_epochs = args.epochs
    start_epoch = args.epochs_start

    best_val = None
    best_params = bundle.params
    train_loss = 0.0

    for epoch_offset in range(num_epochs):
        epoch = start_epoch + epoch_offset

        updated_params, opt_state, train_loss = _train_loop(
            bundle.params,
            optimizer,
            opt_state,
            train_loader,
            loss_fn,
        )
        bundle = ModelBundle(
            config=bundle.config, params=updated_params, module=bundle.module
        )

        val_loss = _evaluate_loop(bundle.params, loss_fn, valid_loader)

        logger.log(
            1,
            f'Epoch {epoch}: train_loss={train_loss:.6f}'
            + (f', val_loss={val_loss:.6f}' if val_loss is not None else ''),
        )

        if val_loss is None:
            best_params = bundle.params
        elif best_val is None or val_loss < best_val:
            best_val = val_loss
            best_params = bundle.params

    _save_parameters(Path(args.output_dir), best_params)

    return {'train_loss': train_loss, 'val_loss': best_val}


def _save_parameters(output_dir: Path, variables) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = output_dir / 'jax_params.msgpack'
    params_path.write_bytes(serialization.to_bytes(variables))


def evaluate(args):
    from . import jax_evaluate as _jax_evaluate

    return _jax_evaluate.evaluate(args)
