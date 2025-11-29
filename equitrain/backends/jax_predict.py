from __future__ import annotations

import itertools

import jax
import numpy as np

from equitrain.argparser import check_args_complete
from equitrain.backends.jax_utils import (
    batched_iterator,
    iter_micro_batches,
    replicate_to_local_devices,
    split_device_outputs,
    stack_or_none,
    supports_multiprocessing_workers,
    take_chunk,
)
from equitrain.backends.jax_utils import is_multi_device as _is_multi_device
from equitrain.backends.jax_utils import (
    prepare_sharded_batch as _prepare_sharded_batch,
)
from equitrain.backends.jax_utils import (
    prepare_single_batch as _prepare_single_batch,
)
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_jax import get_dataloader, make_apply_fn


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
    multi_device = _is_multi_device()
    device_count = jax.local_device_count() if multi_device else 1
    requested_workers = max(int(getattr(args, 'num_workers', 0) or 0), 0)
    if requested_workers > 0 and supports_multiprocessing_workers():
        effective_workers = requested_workers
        if multi_device and device_count > 1:
            effective_workers *= device_count
    else:
        effective_workers = 0
    prefetch_requested = getattr(args, 'prefetch_batches', None)
    if prefetch_requested is None:
        prefetch_batches = effective_workers
    else:
        prefetch_batches = max(int(prefetch_requested or 0), 0)

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
        prefetch_batches=prefetch_batches,
        num_workers=effective_workers,
        graph_multiple=device_count if multi_device else 1,
    )
    if loader is None:
        raise RuntimeError('Prediction dataset is empty.')

    wrapper = _create_wrapper(
        bundle,
        compute_force=getattr(args, 'forces_weight', 0.0) > 0.0,
        compute_stress=getattr(args, 'stress_weight', 0.0) > 0.0,
    )
    base_apply = make_apply_fn(wrapper, num_species=len(z_table))

    use_pmap = multi_device and device_count > 1
    jit_apply = jax.jit(base_apply)
    pmap_apply = None
    params_for_apply = bundle.params
    if use_pmap:
        pmap_apply = jax.pmap(base_apply, axis_name='devices')
        params_for_apply = replicate_to_local_devices(bundle.params)

    energies: list[np.ndarray] = []
    forces: list[np.ndarray] = []
    stresses: list[np.ndarray] = []

    micro_iter = iter_micro_batches(loader)
    group_size = device_count if use_pmap else 1

    first_chunk = take_chunk(micro_iter, group_size)
    if use_pmap and len(first_chunk) < group_size:
        print('[JAX] Disabling multi-GPU prediction: insufficient micro-batches.')
        micro_iter = itertools.chain(first_chunk, micro_iter)
        use_pmap = False
        device_count = 1
        group_size = 1
        params_for_apply = bundle.params
        first_chunk = take_chunk(micro_iter, group_size)

    def _chunk_iterator():
        if first_chunk:
            yield first_chunk
        yield from batched_iterator(
            micro_iter,
            group_size,
            remainder_action=lambda have, need: print(
                f'[JAX] Dropping incomplete multi-device chunk ({have}/{need}).'
            ),
        )

    apply_fn = pmap_apply if use_pmap and pmap_apply is not None else jit_apply

    for chunk in _chunk_iterator():
        graphs = [g for g in chunk if g is not None]
        if not graphs:
            continue

        if use_pmap and pmap_apply is not None:
            if len(graphs) < device_count:
                print(
                    f'[JAX] Skipping multi-device chunk with only {len(graphs)} '
                    f'micro-batches (expected {device_count}).'
                )
                continue
            prepared = _prepare_sharded_batch(graphs, device_count)
            outputs = apply_fn(params_for_apply, prepared)
            device_outputs = split_device_outputs(jax.device_get(outputs), device_count)
        else:
            prepared = _prepare_single_batch(graphs[0])
            outputs = apply_fn(params_for_apply, prepared)
            device_outputs = [jax.device_get(outputs)]

        for result in device_outputs:
            energy_pred = np.asarray(result['energy'])
            energies.append(energy_pred.reshape(-1))

            if result.get('forces') is not None:
                forces.append(np.asarray(result['forces']))
            if result.get('stress') is not None:
                stresses.append(np.asarray(result['stress']))

    return stack_or_none(energies), stack_or_none(forces), stack_or_none(stresses)


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
