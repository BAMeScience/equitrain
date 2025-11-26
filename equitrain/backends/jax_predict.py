from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util as jtu

from equitrain.argparser import check_args_complete
from equitrain.backends.jax_utils import (
    prepare_sharded_batch as _prepare_sharded_batch,
)
from equitrain.backends.jax_utils import (
    prepare_single_batch as _prepare_single_batch,
)
from equitrain.backends.jax_utils import (
    replicate_to_local_devices,
)
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_jax import get_dataloader, make_apply_fn


def _is_multi_device() -> bool:
    return jax.local_device_count() > 1


def _stack_or_none(chunks):
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def _split_device_outputs(tree, num_devices: int):
    host_tree = jtu.tree_map(
        lambda x: None if x is None else np.asarray(x),
        tree,
        is_leaf=lambda leaf: leaf is None,
    )
    slices = []
    for idx in range(num_devices):
        slices.append(
            jtu.tree_map(
                lambda x: None if x is None else x[idx],
                host_tree,
                is_leaf=lambda leaf: leaf is None,
            )
        )
    return slices


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

    predict_path = args.predict_file
    if not (
        predict_path.lower().endswith('.h5') or predict_path.lower().endswith('hdf5')
    ):
        raise ValueError(
            'JAX prediction requires datasets stored in HDF5 format. '
            f'Received: {predict_path}'
        )
    loader = get_dataloader(
        data_file=predict_path,
        atomic_numbers=z_table,
        r_max=r_max,
        batch_size=args.batch_size,
        shuffle=False,
        max_nodes=args.batch_max_nodes,
        max_edges=args.batch_max_edges,
        drop=getattr(args, 'batch_drop', False),
        niggli_reduce=getattr(args, 'niggli_reduce', False),
    )
    if loader is None:
        raise RuntimeError('Prediction dataset is empty.')

    wrapper = _create_wrapper(
        bundle,
        compute_force=getattr(args, 'forces_weight', 0.0) > 0.0,
        compute_stress=getattr(args, 'stress_weight', 0.0) > 0.0,
    )
    base_apply = make_apply_fn(wrapper, num_species=len(z_table))

    multi_device = _is_multi_device()
    device_count = jax.local_device_count() if multi_device else 1
    if multi_device and device_count > 0 and (args.batch_size % device_count != 0):
        raise ValueError(
            'For JAX multi-device prediction, --batch-size must be divisible by the number of local devices.'
        )

    if multi_device and device_count > 1:
        apply_fn = jax.pmap(base_apply, axis_name='devices')
        params_for_apply = replicate_to_local_devices(bundle.params)
    else:
        apply_fn = jax.jit(base_apply)
        params_for_apply = bundle.params

    energies: list[np.ndarray] = []
    forces: list[np.ndarray] = []
    stresses: list[np.ndarray] = []

    for batch in loader:
        micro_batches = batch if isinstance(batch, list) else [batch]
        for micro in micro_batches:
            if multi_device and device_count > 1:
                prepared = _prepare_sharded_batch(micro, device_count)
            else:
                prepared = _prepare_single_batch(micro)

            outputs = apply_fn(params_for_apply, prepared)
            if multi_device and device_count > 1:
                device_outputs = _split_device_outputs(
                    jax.device_get(outputs), device_count
                )
            else:
                device_outputs = [jax.device_get(outputs)]

            for result in device_outputs:
                energy_pred = np.asarray(result['energy'])
                energies.append(energy_pred.reshape(-1))

                if result.get('forces') is not None:
                    forces.append(np.asarray(result['forces']))
                if result.get('stress') is not None:
                    stresses.append(np.asarray(result['stress']))

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
