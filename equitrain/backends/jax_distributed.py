"""Coordinate fixed-shape graph batches across JAX processes."""

from __future__ import annotations

import hashlib
import pickle

import jax
import numpy as np
from jax.experimental import multihost_utils

from equitrain.backends.jax_utils import (
    _pad_graphs_for_device_stack,
    iter_micro_batches,
    take_chunk,
)


def _set_weight(graph, weight):
    globals_ = graph.globals
    if hasattr(globals_, '_replace'):
        globals_ = globals_._replace(weight=weight)
    else:
        globals_ = type(globals_)(globals_) if globals_ is not None else {}
        globals_['weight'] = weight
    return graph._replace(globals=globals_)


def _with_weight(graph):
    globals_ = graph.globals
    weight = (
        globals_.get('weight')
        if isinstance(globals_, dict)
        else getattr(globals_, 'weight', None)
    )
    if weight is None:
        return _set_weight(graph, np.ones(graph.n_node.shape, dtype=np.float32))
    return graph


def empty_graph_like(graph):
    """Keep valid geometry and topology; exclude every graph through its weight."""
    weight = (
        graph.globals.get('weight')
        if isinstance(graph.globals, dict)
        else getattr(graph.globals, 'weight', None)
    )
    dtype = np.float32 if weight is None else weight.dtype
    return _set_weight(graph, np.zeros(graph.n_node.shape, dtype=dtype))


def _signature(graph):
    leaves, tree = jax.tree_util.tree_flatten(graph)
    schema = (str(tree), [(x.shape, str(x.dtype)) for x in leaves])
    return np.frombuffer(hashlib.sha256(repr(schema).encode()).digest(), dtype=np.int32)


def _broadcast_template(graph, source):
    is_source = jax.process_index() == source
    payload = pickle.dumps(jax.device_get(graph)) if is_source else b''
    size = multihost_utils.broadcast_one_to_all(
        np.asarray(len(payload), dtype=np.int32), is_source=is_source
    )
    buffer = (
        np.frombuffer(payload, dtype=np.uint8)
        if is_source
        else np.zeros(int(size), dtype=np.uint8)
    )
    buffer = multihost_utils.broadcast_one_to_all(buffer, is_source=is_source)
    return pickle.loads(np.asarray(buffer).tobytes())


def coordinated_chunks(loader, device_count, *, phase, logger, sync_processes=False):
    """Keep all ranks participating until every loader is exhausted.

    Run on the main thread so loader coordination cannot race model collectives.
    Empty ranks receive a valid first batch as their zero-weight padding template.
    """
    if device_count < 1:
        raise ValueError('device_count must be positive.')
    micro_iter = iter_micro_batches(loader)
    template = None
    first_step = True
    warned = False
    while True:
        error = None
        try:
            chunk = [_with_weight(g) for g in take_chunk(micro_iter, device_count)]
            if chunk:
                chunk = _pad_graphs_for_device_stack(chunk)
                template = chunk[0]
            signature = None if template is None else _signature(template)
            if any(not np.array_equal(_signature(g), signature) for g in chunk):
                raise ValueError(
                    'Local micro-batches must have matching shapes and dtypes.'
                )
        except Exception as exc:
            error = exc
            chunk = []
            signature = None

        if sync_processes:
            state = np.zeros(9, dtype=np.int32)
            state[0] = -1 if error is not None else int(bool(chunk))
            if signature is not None:
                state[1:] = signature
            states = np.asarray(multihost_utils.process_allgather(state, tiled=False))
            failed = np.flatnonzero(states[:, 0] < 0)
            if failed.size:
                raise RuntimeError(
                    f'[{phase}] Batch loading failed on processes {failed.tolist()}.'
                ) from error
            active = np.flatnonzero(states[:, 0] > 0)
            if not active.size:
                return
            signatures = states[np.any(states[:, 1:] != 0, axis=1), 1:]
            if not np.all(signatures == signatures[0]):
                raise ValueError(
                    f'[{phase}] Processes must use matching fixed batch shapes and dtypes.'
                )
            if first_step and active.size < jax.process_count():
                shared = _broadcast_template(template, int(active[0]))
                if template is None:
                    template = shared
        else:
            if error is not None:
                raise error
            if not chunk:
                return
        first_step = False
        if len(chunk) < device_count:
            if logger is not None and not warned:
                logger.log(
                    1,
                    f'[{phase}] Padding incomplete device chunk ({len(chunk)}/{device_count}).',
                )
                warned = True
            chunk.extend(
                empty_graph_like(template) for _ in range(device_count - len(chunk))
            )
        yield chunk
