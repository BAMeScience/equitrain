"""Utility helpers for JAX backends (model loading)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jraph
import numpy as np
from flax import core as flax_core
from flax import serialization
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
    build_module = get_wrapper_builder(wrapper_name)

    jax_module, template = build_module(config)
    variables = jax_module.init(jax.random.PRNGKey(0), template)
    variables = serialization.from_bytes(variables, params_path.read_bytes())
    variables = flax_core.freeze(variables)

    return ModelBundle(config=config, params=variables, module=jax_module)


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
    return [
        _slice_graph(graph, i * per_device, per_device) for i in range(num_devices)
    ]


def prepare_sharded_batch(graph, num_devices: int):
    """Prepare a micro-batch for ``jax.pmap`` execution."""
    chunks = split_graphs_for_devices(graph, num_devices)
    device_batches = []
    for chunk in chunks:
        if isinstance(chunk, jraph.GraphsTuple):
            graphs_tuple = chunk
        else:
            graphs_tuple = chunk[0] if len(chunk) == 1 else jraph.batch_np(chunk)
        device_batches.append(prepare_single_batch(graphs_tuple))

    def _stack_or_none(*values):
        first = values[0]
        if first is None:
            return None
        return jnp.stack(values)

    return jtu.tree_map(_stack_or_none, *device_batches, is_leaf=_none_leaf)


__all__ = [
    'ModelBundle',
    'set_jax_dtype',
    'resolve_model_paths',
    'load_model_bundle',
    'replicate_to_local_devices',
    'unreplicate_from_local_devices',
    'prepare_single_batch',
    'prepare_sharded_batch',
    'split_graphs_for_devices',
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
