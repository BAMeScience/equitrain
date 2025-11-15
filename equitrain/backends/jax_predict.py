from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util as jtu

from equitrain.argparser import check_args_complete
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_jax import atoms_to_graphs, build_loader, make_apply_fn


def _is_multi_device() -> bool:
    return jax.local_device_count() > 1


def _prepare_single_batch(graph):
    def _to_device_array(x):
        if x is None:
            return None
        return jnp.asarray(x)

    return jtu.tree_map(_to_device_array, graph, is_leaf=lambda leaf: leaf is None)


def _stack_or_none(chunks):
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def predict(args):
    check_args_complete(args, 'predict')
    backend = getattr(args, 'backend', 'torch') or 'torch'
    if backend != 'jax':
        raise NotImplementedError(
            f'JAX predict backend invoked with unsupported backend="{backend}".'
        )

    if getattr(args, 'predict_file', None) is None:
        raise ValueError('--predict-file is a required argument for JAX prediction.')
    if getattr(args, 'model', None) is None:
        raise ValueError('--model is a required argument for JAX prediction.')

    if _is_multi_device():
        raise NotImplementedError(
            'JAX prediction currently supports single-device runs only. '
            'Set XLA flags to limit execution to one device.'
        )

    bundle = _load_bundle(
        args.model,
        dtype=args.dtype,
        wrapper=getattr(args, 'model_wrapper', None),
    )

    atomic_numbers = bundle.config.get('atomic_numbers')
    if not atomic_numbers:
        raise RuntimeError('Model configuration is missing `atomic_numbers`.')
    z_table = AtomicNumberTable(list(atomic_numbers))

    r_max = (
        float(args.r_max)
        if getattr(args, 'r_max', None)
        else float(bundle.config.get('r_max', 0.0))
    )
    if r_max <= 0.0:
        raise RuntimeError(
            'Model configuration must define a positive `r_max`, or override via --r-max.'
        )

    graphs = atoms_to_graphs(
        args.predict_file,
        r_max,
        z_table,
        niggli_reduce=getattr(args, 'niggli_reduce', False),
    )
    loader = build_loader(
        graphs,
        batch_size=args.batch_size,
        shuffle=False,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
        drop=getattr(args, 'batch_drop', False),
    )
    if loader is None:
        raise RuntimeError('Prediction dataset is empty.')

    wrapper = _create_wrapper(
        bundle,
        compute_force=getattr(args, 'forces_weight', 0.0) > 0.0,
        compute_stress=getattr(args, 'stress_weight', 0.0) > 0.0,
    )
    apply_fn = make_apply_fn(wrapper, num_species=len(z_table))
    apply_fn = jax.jit(apply_fn)

    energies: list[np.ndarray] = []
    forces: list[np.ndarray] = []
    stresses: list[np.ndarray] = []

    for batch in loader:
        micro_batches = batch if isinstance(batch, list) else [batch]
        for micro in micro_batches:
            prepared = _prepare_single_batch(micro)
            outputs = jax.device_get(apply_fn(bundle.params, prepared))
            energy_pred = np.asarray(outputs['energy'])
            energies.append(energy_pred.reshape(-1))

            if outputs.get('forces') is not None:
                forces.append(np.asarray(outputs['forces']))
            if outputs.get('stress') is not None:
                stresses.append(np.asarray(outputs['stress']))

    return _stack_or_none(energies), _stack_or_none(forces), _stack_or_none(stresses)


def _load_bundle(model_path: str, dtype: str, wrapper: str | None):
    from equitrain.backends.jax_utils import load_model_bundle as _load_model_bundle

    return _load_model_bundle(model_path, dtype=dtype, wrapper=wrapper)


def _create_wrapper(bundle, *, compute_force: bool, compute_stress: bool):
    from equitrain.backends.jax_wrappers import MaceWrapper as JaxMaceWrapper

    return JaxMaceWrapper(
        module=bundle.module,
        config=bundle.config,
        compute_force=compute_force,
        compute_stress=compute_stress,
    )


__all__ = ['predict']
