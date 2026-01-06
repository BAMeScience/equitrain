from __future__ import annotations

import functools
import logging
import pickle
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path
from queue import Queue
from typing import Any

import jax
import jraph
import numpy as np
from tqdm import tqdm

from equitrain.argparser import check_args_complete
from equitrain.backends.jax_utils import (
    prepare_sharded_batch,
    set_jax_platform,
    split_graphs_for_devices,
    stack_or_none,
    supports_multiprocessing_workers,
)
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_jax import get_dataloader, make_apply_fn
from equitrain.data.backend_jax.loaders_impl_cache import (
    STREAMING_STATS_CACHE_VERSION,
    dataset_signature,
    stats_cache_path,
    stats_payload_to_parts,
)

_GRAPH_OUTPUT_KEYS = {
    'energy',
    'stress',
    'virials',
    'dipole',
}
_NODE_OUTPUT_KEYS = {
    'forces',
}
_EDGE_OUTPUT_KEYS: set[str] = set()
_SKIP_OUTPUT_KEYS: set[str] = {
    'lammps_natoms',
}


def _prefetch_to_device(iterator, capacity: int, device_put_fn: Callable[[Any], Any]):
    """Prefetch items onto device(s) using a bounded queue."""
    if capacity is None or capacity <= 0:
        return iterator
    queue: Queue = Queue(maxsize=capacity)
    sentinel = object()
    total_hint = getattr(iterator, 'total_batches_hint', 0)

    def _producer():
        try:
            for item in iterator:
                queue.put(device_put_fn(item))
        finally:
            queue.put(sentinel)

    threading.Thread(target=_producer, daemon=True).start()

    class _Prefetched:
        def __init__(self):
            self._done = False
            self.total_batches_hint = total_hint

        def __iter__(self):
            return self

        def __next__(self):
            if self._done:
                raise StopIteration
            item = queue.get()
            if item is sentinel:
                self._done = True
                raise StopIteration
            return item

    return _Prefetched()


def _filter_by_mask(values: list[Any], mask: np.ndarray) -> list[Any]:
    if len(values) != len(mask):
        raise ValueError(
            f'Mismatched graph mask length ({len(values)} values vs {len(mask)} mask).'
        )
    return [value for value, keep in zip(values, mask) if keep]


def _split_prediction_outputs(
    outputs: dict[str, Any],
    graph: jraph.GraphsTuple,
) -> tuple[list[int], dict[str, list[Any]]]:
    graph_ids = getattr(graph.globals, 'graph_id', None)
    if graph_ids is None:
        raise ValueError(
            'Streaming prediction requires graph.globals.graph_id to be set.'
        )
    graph_ids = np.asarray(graph_ids).reshape(-1)
    graph_mask = np.asarray(jraph.get_graph_padding_mask(graph), dtype=bool)
    if graph_ids.shape[0] != graph_mask.shape[0]:
        raise ValueError(
            'graph_id length does not match graph batch size '
            f'({graph_ids.shape[0]} vs {graph_mask.shape[0]}).'
        )
    valid_mask = graph_mask & (graph_ids >= 0)

    n_node = np.asarray(graph.n_node, dtype=int)
    n_edge = np.asarray(graph.n_edge, dtype=int)
    total_nodes = int(n_node.sum())
    total_edges = int(n_edge.sum())
    n_graphs = int(n_node.shape[0])

    def _split_by_counts(arr: np.ndarray, counts: np.ndarray) -> list[np.ndarray]:
        if counts.size == 0:
            return []
        total = int(counts.sum())
        if total == 0:
            return [arr[:0]] * len(counts)
        splits = np.cumsum(counts)[:-1]
        return np.split(arr[:total], splits, axis=0)

    per_graph_outputs: dict[str, list[Any]] = {}
    for key, value in outputs.items():
        if key in _SKIP_OUTPUT_KEYS:
            continue
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim == 0:
            continue
        if key in _NODE_OUTPUT_KEYS:
            chunks = _split_by_counts(arr, n_node)
            per_graph_outputs[key] = _filter_by_mask(chunks, valid_mask)
            continue
        if key in _EDGE_OUTPUT_KEYS:
            chunks = _split_by_counts(arr, n_edge)
            per_graph_outputs[key] = _filter_by_mask(chunks, valid_mask)
            continue
        if key in _GRAPH_OUTPUT_KEYS:
            arr = arr[:n_graphs]
            per_graph_outputs[key] = _filter_by_mask(list(arr), valid_mask)
            continue

        if arr.shape[0] >= total_nodes > 0:
            chunks = _split_by_counts(arr, n_node)
            per_graph_outputs[key] = _filter_by_mask(chunks, valid_mask)
        elif arr.shape[0] >= total_edges > 0:
            chunks = _split_by_counts(arr, n_edge)
            per_graph_outputs[key] = _filter_by_mask(chunks, valid_mask)
        elif arr.shape[0] >= n_graphs:
            arr = arr[:n_graphs]
            per_graph_outputs[key] = _filter_by_mask(list(arr), valid_mask)
        else:
            raise ValueError(
                f'Output {key} has incompatible leading dimension {arr.shape[0]} '
                f'(graphs={n_graphs}, nodes={total_nodes}, edges={total_edges}).'
            )

    graph_ids = [int(val) for val in graph_ids[valid_mask]]
    return graph_ids, per_graph_outputs


def predict_streaming(
    predictor: Callable,
    params: Any,
    data_loader,
    name: str = 'Prediction',
    progress_bar: bool = True,
    device_prefetch_batches: int | None = None,
) -> tuple[list[int], dict[str, list[Any]]]:
    """Run the predictor on a streaming loader and order outputs by graph_id."""
    local_devices = jax.local_devices()
    local_device_count = max(1, len(local_devices))
    process_index = getattr(jax, 'process_index', lambda: 0)()
    process_count = getattr(jax, 'process_count', lambda: 1)()
    is_primary = process_index == 0

    iterator = data_loader.iter_batches(
        epoch=0,
        seed=None,
        process_count=process_count,
        process_index=process_index,
    )
    total_hint = getattr(iterator, 'total_batches_hint', 0)
    if not total_hint:
        approx_length = getattr(data_loader, 'approx_length', None)
        if callable(approx_length):
            try:
                total_hint = int(approx_length())
            except Exception:  # pragma: no cover - defensive fallback
                total_hint = 0

    _predict_step = functools.partial(
        jax.pmap,
        in_axes=(None, 0),
        out_axes=0,
        axis_name='devices',
        devices=local_devices,
    )(predictor)

    def _prepare_device_graphs(graph):
        if local_device_count <= 1:
            if graph.n_node.ndim == 1:
                device_batch = jax.tree_util.tree_map(lambda x: x[None, ...], graph)
            else:
                device_batch = graph
            return device_batch, [graph]
        if graph.n_node.ndim == 1:
            device_batch = prepare_sharded_batch(graph, local_device_count)
        else:
            if graph.n_node.shape[0] != local_device_count:
                raise ValueError(
                    'Expected microbatches with leading axis equal to the number of '
                    f'local devices ({local_device_count}), got axis size '
                    f'{graph.n_node.shape[0]}.'
                )
            device_batch = graph
        device_graphs = split_graphs_for_devices(graph, local_device_count)
        return device_batch, device_graphs

    host_prefetch_cap = int(getattr(data_loader, '_prefetch_batches', 0) or 0)
    device_prefetch_cap = device_prefetch_batches
    if device_prefetch_cap is None:
        device_prefetch_cap = max(host_prefetch_cap, 2)
    device_prefetch_cap = int(device_prefetch_cap or 0)
    device_prefetch_active = device_prefetch_cap > 0
    if device_prefetch_active:
        source_iter = iterator

        def _device_put(graph):
            device_batch, device_graphs = _prepare_device_graphs(graph)
            device_batch = jax.device_put(device_batch)
            return device_batch, device_graphs

        iterator = _prefetch_to_device(source_iter, device_prefetch_cap, _device_put)

    p_bar = tqdm(
        iterator,
        desc=name,
        total=total_hint or None,
        disable=not (progress_bar and is_primary),
    )

    def _select_device_output(value, device_idx):
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] == local_device_count:
            return value[device_idx]
        return value

    def _process_outputs(device_outputs, device_graphs):
        host_outputs = jax.device_get(device_outputs)
        if local_device_count <= 1:
            host_outputs = jax.tree_util.tree_map(
                lambda x: _select_device_output(x, 0),
                host_outputs,
                is_leaf=lambda x: x is None,
            )
            return _split_prediction_outputs(host_outputs, device_graphs[0])
        batch_graph_ids: list[int] = []
        batch_outputs: dict[str, list[Any]] = {}
        for device_idx, device_graph in enumerate(device_graphs):
            device_outputs = jax.tree_util.tree_map(
                lambda x: _select_device_output(x, device_idx),
                host_outputs,
                is_leaf=lambda x: x is None,
            )
            ids, per_graph = _split_prediction_outputs(device_outputs, device_graph)
            for key in list(batch_outputs):
                if key not in per_graph:
                    batch_outputs[key].extend([None] * len(ids))
            for key, values in per_graph.items():
                if key not in batch_outputs:
                    batch_outputs[key] = [None] * len(batch_graph_ids)
                batch_outputs[key].extend(values)
            batch_graph_ids.extend(ids)
        return batch_graph_ids, batch_outputs

    def _accumulate(batch_graph_ids, batch_outputs, graph_ids, outputs):
        for key in list(outputs):
            if key not in batch_outputs:
                outputs[key].extend([None] * len(batch_graph_ids))
        for key, values in batch_outputs.items():
            if key not in outputs:
                outputs[key] = [None] * len(graph_ids)
            outputs[key].extend(values)
        graph_ids.extend(batch_graph_ids)

    graph_ids: list[int] = []
    outputs: dict[str, list[Any]] = {}
    pending: deque[tuple[Any, list[jraph.GraphsTuple]]] = deque()
    output_prefetch_cap = max(int(device_prefetch_cap or 0), 1)

    for item in p_bar:
        if device_prefetch_active:
            device_batch, device_graphs = item
        else:
            device_batch, device_graphs = _prepare_device_graphs(item)
        raw_outputs = _predict_step(params, device_batch)
        pending.append((raw_outputs, device_graphs))
        if len(pending) > output_prefetch_cap:
            raw_outputs, device_graphs = pending.popleft()
            batch_graph_ids, batch_outputs = _process_outputs(
                raw_outputs, device_graphs
            )
            _accumulate(batch_graph_ids, batch_outputs, graph_ids, outputs)

    while pending:
        raw_outputs, device_graphs = pending.popleft()
        batch_graph_ids, batch_outputs = _process_outputs(raw_outputs, device_graphs)
        _accumulate(batch_graph_ids, batch_outputs, graph_ids, outputs)

    if p_bar.total is not None and p_bar.n != p_bar.total:
        p_bar.total = p_bar.n
        p_bar.refresh()
    p_bar.close()

    if not graph_ids:
        return [], {}

    order = np.argsort(np.asarray(graph_ids))
    ordered_graph_ids = [graph_ids[idx] for idx in order]
    ordered_outputs = {
        key: [values[idx] for idx in order] for key, values in outputs.items()
    }
    return ordered_graph_ids, ordered_outputs


def _load_cached_streaming_caps(path):
    cache_path = stats_cache_path(path)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open('rb') as fh:
            payload = pickle.load(fh)
    except Exception:  # pragma: no cover - cache corruption is unexpected
        return None
    if payload is None or payload.get('version') != STREAMING_STATS_CACHE_VERSION:
        return None
    if payload.get('dataset_signature') != dataset_signature(path):
        return None
    stats_payload = payload.get('stats')
    if not stats_payload:
        return None
    try:
        n_nodes, n_edges, n_graphs, _ = stats_payload_to_parts(stats_payload)
    except (KeyError, TypeError, ValueError):
        return None
    return n_nodes, n_edges, n_graphs


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

    set_jax_platform(getattr(args, 'jax_platform', None))

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

    batch_max_edges = getattr(args, 'batch_max_edges', None)
    batch_max_nodes = getattr(args, 'batch_max_nodes', None)
    if batch_max_edges is None:
        cached_caps = _load_cached_streaming_caps(Path(predict_path))
        if cached_caps is None:
            raise ValueError(
                'JAX prediction requires --batch-max-edges unless cached streaming '
                'stats are available.'
            )
        cached_nodes, cached_edges, cached_graphs = cached_caps
        batch_max_edges = cached_edges
        if batch_max_nodes is None:
            batch_max_nodes = cached_nodes
        logging.info(
            'Using cached streaming stats for %s: n_nodes=%s n_edges=%s n_graphs=%s',
            predict_path,
            cached_nodes,
            cached_edges,
            cached_graphs,
        )

    requested_workers = max(int(getattr(args, 'num_workers', 0) or 0), 0)
    if requested_workers > 0 and supports_multiprocessing_workers():
        effective_workers = requested_workers
        device_count = jax.local_device_count()
        if device_count > 1:
            effective_workers *= device_count
    else:
        effective_workers = 0

    prefetch_batches = getattr(args, 'prefetch_batches', None)
    loader = get_dataloader(
        data_file=predict_path,
        atomic_numbers=z_table,
        r_max=r_max,
        shuffle=False,
        max_nodes=batch_max_nodes,
        max_edges=batch_max_edges,
        drop=getattr(args, 'batch_drop', False),
        niggli_reduce=getattr(args, 'niggli_reduce', False),
        prefetch_batches=prefetch_batches,
        num_workers=effective_workers,
        graph_multiple=None,
    )
    if loader is None:
        raise RuntimeError('Prediction dataset is empty.')
    effective_nodes = getattr(loader, '_n_node', batch_max_nodes)
    effective_edges = getattr(loader, '_n_edge', batch_max_edges)
    if batch_max_edges is not None and effective_edges != batch_max_edges:
        logging.warning(
            'Requested max edges per batch (%s) was raised to %s to fit the data.',
            batch_max_edges,
            effective_edges,
        )
    if batch_max_nodes is not None and effective_nodes != batch_max_nodes:
        logging.warning(
            'Requested max nodes per batch (%s) was raised to %s to fit the data.',
            batch_max_nodes,
            effective_nodes,
        )

    wrapper = _create_wrapper(
        bundle,
        compute_force=getattr(args, 'forces_weight', 0.0) > 0.0,
        compute_stress=getattr(args, 'stress_weight', 0.0) > 0.0,
    )
    base_apply = make_apply_fn(wrapper, num_species=len(z_table))

    tqdm_desc = getattr(args, 'tqdm_desc', None)
    graph_ids, outputs = predict_streaming(
        base_apply,
        bundle.params,
        loader,
        name=str(tqdm_desc) if tqdm_desc else 'JAX predict',
        progress_bar=getattr(args, 'tqdm', False),
    )

    if not graph_ids or not outputs:
        return None, None, None

    energies = np.asarray(outputs['energy']) if 'energy' in outputs else None
    forces = stack_or_none(outputs.get('forces')) if 'forces' in outputs else None
    stresses = np.asarray(outputs['stress']) if 'stress' in outputs else None
    return energies, forces, stresses


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
