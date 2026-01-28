"""Utility helpers for JAX backends (model loading)."""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jraph
import numpy as np
from flax import nnx, serialization
from jax import tree_util as jtu

from equitrain.argparser import ArgumentError
from equitrain.backends.jax_wrappers import get_wrapper_builder

DEFAULT_CONFIG_NAME = 'config.json'
DEFAULT_PARAMS_NAME = 'params.msgpack'


@dataclass(frozen=True)
class ModelBundle:
    config: dict
    params: dict
    module: object


def set_jax_dtype(dtype: str) -> None:
    dtype = (dtype or 'float32').lower()
    if dtype == 'float64':
        jax.config.update('jax_enable_x64', True)
    elif dtype in {'float32', 'float16'}:
        jax.config.update('jax_enable_x64', False)
    else:
        raise ArgumentError(f'Unsupported dtype for JAX backend: {dtype}')


def set_jax_platform(platform: str | None) -> None:
    if not platform:
        return
    normalized = platform.strip().lower()
    if normalized in {'auto', 'default', ''}:
        return
    if normalized not in {'cpu', 'gpu', 'tpu'}:
        raise ArgumentError(
            f'Unsupported JAX platform "{platform}". Expected one of cpu/gpu/tpu.'
        )
    jax.config.update('jax_platform_name', normalized)


def resolve_model_paths(model_arg: str) -> tuple[Path, Path]:
    path = Path(model_arg).expanduser().resolve()

    if path.is_dir():
        config_path = path / DEFAULT_CONFIG_NAME
        params_path = path / DEFAULT_PARAMS_NAME
    elif path.suffix == '.json':
        config_path = path
        params_path = path.with_suffix('.msgpack')
    else:
        params_path = path
        config_path = path.with_suffix('.json')

    if not config_path.exists():
        raise FileNotFoundError(
            f'Unable to locate JAX model configuration at {config_path}'
        )
    if not params_path.exists():
        raise FileNotFoundError(
            f'Unable to locate serialized JAX parameters at {params_path}'
        )

    return config_path, params_path


def _discover_wrapper_name(config: dict, explicit: str | None) -> str:
    if explicit:
        return explicit.strip().lower()

    for key in ('model_wrapper', 'wrapper', 'wrapper_name'):
        value = config.get(key)
        if value:
            return str(value).strip().lower()
    return 'mace'


def load_model_bundle(
    model_arg: str,
    dtype: str,
    *,
    wrapper: str | None = None,
) -> ModelBundle:
    config_path, params_path = resolve_model_paths(model_arg)
    config = json.loads(config_path.read_text())

    set_jax_dtype(dtype)

    wrapper_name = _discover_wrapper_name(config, wrapper)

    if wrapper_name == 'mace':
        from mace_jax.tools import bundle as mace_bundle  # noqa: PLC0415

        bundle = mace_bundle.load_model_bundle(
            model_arg,
            dtype=dtype,
            wrapper='mace',
        )
        return ModelBundle(
            config=bundle.config,
            params=bundle.params,
            module=bundle.graphdef,
        )

    build_module = get_wrapper_builder(wrapper_name)

    jax_module, template = build_module(config)
    graphdef, state = nnx.split(jax_module)
    from mace_jax.nnx_utils import (  # noqa: PLC0415
        state_to_pure_dict,
        state_to_serializable_dict,
    )

    state_template = state_to_serializable_dict(state)
    state_pure = serialization.from_bytes(state_template, params_path.read_bytes())
    nnx.replace_by_pure_dict(state, state_pure)
    params = state_to_pure_dict(state)

    return ModelBundle(config=config, params=params, module=graphdef)


def is_multi_device() -> bool:
    """Return ``True`` if more than one local JAX device is available."""
    return jax.local_device_count() > 1


def _none_leaf(value):
    return value is None


def replicate_to_local_devices(tree):
    """Broadcast a pytree so the leading axis matches local device count."""
    device_count = jax.local_device_count()
    if device_count <= 1:
        return tree

    def _replicate(leaf):
        if leaf is None:
            return None
        arr = jnp.asarray(leaf)
        broadcast = jnp.broadcast_to(arr, (device_count,) + arr.shape)
        return broadcast

    return jtu.tree_map(_replicate, tree, is_leaf=_none_leaf)


def unreplicate_from_local_devices(tree):
    """Strip a replicated leading axis (if present) from a pytree."""
    device_count = jax.local_device_count()
    if device_count <= 1:
        return tree

    host = jax.device_get(tree)
    if isinstance(host, list | tuple) and len(host) == device_count:
        return jtu.tree_map(lambda x: x[0], host, is_leaf=_none_leaf)

    def _maybe_collapse(leaf):
        if leaf is None:
            return None
        arr = np.asarray(leaf)
        if arr.ndim == 0 or arr.shape[0] != device_count:
            return leaf
        first = arr[0]
        if np.all(arr == first):
            return first
        return leaf

    return jtu.tree_map(_maybe_collapse, host, is_leaf=_none_leaf)


def prepare_single_batch(graph):
    """Cast a batched graph to device arrays, keeping None leaves."""

    def _to_device_array(x):
        if x is None:
            return None
        return jnp.asarray(x)

    return jtu.tree_map(_to_device_array, graph, is_leaf=_none_leaf)


def split_graphs_for_devices(graph, num_devices: int) -> list[jraph.GraphsTuple]:
    def _pad_graphs_to_multiple(graph, multiple):
        if multiple <= 1:
            return graph
        total = int(graph.n_node.shape[0])
        remainder = total % multiple
        if remainder == 0:
            return graph
        pad_graphs = multiple - remainder
        pad_n_node = np.concatenate(
            [np.asarray(graph.n_node), np.zeros(pad_graphs, dtype=np.int32)]
        )
        pad_n_edge = np.concatenate(
            [np.asarray(graph.n_edge), np.zeros(pad_graphs, dtype=np.int32)]
        )

        globals_attr = graph.globals
        if globals_attr is None:
            globals_dict = None
        elif hasattr(globals_attr, 'items'):
            globals_dict = globals_attr.__class__()
            for key, value in globals_attr.items():
                pad_shape = (pad_graphs,) + value.shape[1:]
                pad_vals = np.zeros(pad_shape, dtype=value.dtype)
                globals_dict[key] = np.concatenate([value, pad_vals], axis=0)
        else:
            pad_shape = (pad_graphs,) + globals_attr.shape[1:]
            pad_vals = np.zeros(pad_shape, dtype=globals_attr.dtype)
            globals_dict = np.concatenate([globals_attr, pad_vals], axis=0)

        return graph._replace(
            globals=globals_dict,
            n_node=pad_n_node,
            n_edge=pad_n_edge,
        )

    graph = _pad_graphs_to_multiple(graph, num_devices)
    total_graphs = int(graph.n_node.shape[0])
    if total_graphs % num_devices != 0:
        raise ValueError(
            'For JAX multi-device execution, batch size must be divisible by the number of devices.'
        )
    per_device = total_graphs // num_devices
    return [_slice_graph(graph, i * per_device, per_device) for i in range(num_devices)]


def prepare_sharded_batch(graph, num_devices: int):
    """Prepare a micro-batch for ``jax.pmap`` execution."""

    def _ensure_graphs_tuple(item):
        if isinstance(item, jraph.GraphsTuple):
            return item
        if isinstance(item, Sequence) and not isinstance(item, bytes | str):
            return item[0] if len(item) == 1 else jraph.batch_np(item)
        raise TypeError('Expected a GraphsTuple or sequence of GraphsTuples.')

    device_batches = []
    if isinstance(graph, Sequence) and not isinstance(graph, jraph.GraphsTuple):
        filtered = [g for g in graph if g is not None]
        if len(filtered) != num_devices:
            raise ValueError(
                f'Expected {num_devices} micro-batches for multi-device execution, '
                f'got {len(filtered)}.'
            )
        for item in filtered:
            device_batches.append(prepare_single_batch(_ensure_graphs_tuple(item)))
    else:
        chunks = split_graphs_for_devices(graph, num_devices)
        for chunk in chunks:
            device_batches.append(prepare_single_batch(_ensure_graphs_tuple(chunk)))

    def _stack_or_none(*values):
        first = values[0]
        if first is None:
            return None
        return jnp.stack(values)

    return jtu.tree_map(_stack_or_none, *device_batches, is_leaf=_none_leaf)


_MP_WORKERS_SUPPORTED: bool | None = None


def supports_multiprocessing_workers() -> bool:
    """Return True if this environment can create ``multiprocessing`` locks."""
    global _MP_WORKERS_SUPPORTED
    if _MP_WORKERS_SUPPORTED is not None:
        return _MP_WORKERS_SUPPORTED

    try:
        ctx = mp.get_context('spawn')
        lock = ctx.Lock()
        lock.acquire()
        lock.release()
    except (OSError, PermissionError):
        _MP_WORKERS_SUPPORTED = False
    else:
        _MP_WORKERS_SUPPORTED = True
    return _MP_WORKERS_SUPPORTED


def stack_or_none(chunks):
    """Concatenate numpy arrays unless the list is empty."""
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def split_device_outputs(tree, num_devices: int):
    """Slice a replicated pytree into per-device host arrays."""
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


def iter_micro_batches(loader):
    """Flatten a loader that may emit lists of micro-batches."""
    for item in loader:
        if item is None:
            continue
        if isinstance(item, list):
            for sub in item:
                if sub is not None:
                    yield sub
        else:
            yield item


def take_chunk(iterator, size: int):
    """Collect up to ``size`` items from an iterator."""
    return list(itertools.islice(iterator, size))


def batched_iterator(
    iterator,
    size: int,
    *,
    remainder_action: callable | None = None,
    drop_remainder: bool = True,
):
    """Yield fixed-size chunks from ``iterator``.

    If ``drop_remainder`` is False, the final partial chunk (if any) is yielded
    after ``remainder_action`` is invoked.
    """
    if size <= 0:
        raise ValueError('Chunk size must be positive.')
    while True:
        chunk = take_chunk(iterator, size)
        if len(chunk) < size:
            if chunk and remainder_action is not None:
                remainder_action(len(chunk), size)
            if chunk and not drop_remainder:
                yield chunk
            break
        yield chunk


__all__ = [
    'ModelBundle',
    'set_jax_dtype',
    'set_jax_platform',
    'resolve_model_paths',
    'load_model_bundle',
    'is_multi_device',
    'replicate_to_local_devices',
    'unreplicate_from_local_devices',
    'prepare_single_batch',
    'prepare_sharded_batch',
    'split_graphs_for_devices',
    'stack_or_none',
    'split_device_outputs',
    'iter_micro_batches',
    'take_chunk',
    'batched_iterator',
    'supports_multiprocessing_workers',
]


def _slice_graph(graph: jraph.GraphsTuple, start_graph: int, count: int):
    start_graph = int(start_graph)
    count = int(count)
    n_node = np.asarray(graph.n_node)
    n_edge = np.asarray(graph.n_edge)

    graph_slice = slice(start_graph, start_graph + count)
    node_start = int(n_node[:start_graph].sum())
    node_end = int(node_start + n_node[graph_slice].sum())
    edge_start = int(n_edge[:start_graph].sum())
    edge_end = int(edge_start + n_edge[graph_slice].sum())

    nodes = graph.nodes.__class__()
    for key, value in graph.nodes.items():
        nodes[key] = value[node_start:node_end]

    edges = graph.edges.__class__()
    for key, value in graph.edges.items():
        edges[key] = value[edge_start:edge_end]

    senders = graph.senders[edge_start:edge_end] - node_start
    receivers = graph.receivers[edge_start:edge_end] - node_start

    globals_attr = graph.globals
    if globals_attr is None:
        globals_dict = None
    elif hasattr(globals_attr, 'items'):
        globals_dict = globals_attr.__class__()
        for key, value in globals_attr.items():
            globals_dict[key] = value[graph_slice]
    else:
        globals_dict = globals_attr[graph_slice]
    n_node_slice = graph.n_node[graph_slice]
    n_edge_slice = graph.n_edge[graph_slice]

    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=globals_dict,
        n_node=n_node_slice,
        n_edge=n_edge_slice,
    )
