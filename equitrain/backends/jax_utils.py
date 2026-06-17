"""Utility helpers for JAX backends (model loading)."""

from __future__ import annotations

import inspect
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
from jax.sharding import Mesh, NamedSharding, PartitionSpec

try:  # pragma: no cover - exercised only with older JAX releases
    from jax import shard_map as _shard_map
except (ImportError, AttributeError):  # pragma: no cover
    try:
        from jax.experimental.shard_map import shard_map as _shard_map
    except (ImportError, AttributeError, ModuleNotFoundError):
        _shard_map = None

from equitrain.argparser import ArgumentError
from equitrain.backends.jax_wrappers import get_wrapper_builder, infer_wrapper_name

DEFAULT_CONFIG_NAME = 'config.json'
DEFAULT_PARAMS_NAME = 'params.msgpack'
DEVICE_AXIS_NAME = 'device'
DEVICE_AXIS_SPEC = PartitionSpec(DEVICE_AXIS_NAME)
REPLICATED_SPEC = PartitionSpec()


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


def load_model_bundle(
    model_arg: str,
    dtype: str,
    *,
    wrapper: str | None = None,
) -> ModelBundle:
    config_path, params_path = resolve_model_paths(model_arg)
    config = json.loads(config_path.read_text())

    set_jax_dtype(dtype)

    wrapper_name = infer_wrapper_name(config, wrapper)

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
    if template is not None:
        params_template = template
        module = jax_module
    else:
        try:
            graphdef, state = nnx.split(jax_module)
        except Exception as exc:
            raise TypeError(
                f'JAX wrapper "{wrapper_name}" returned no params template and '
                'the module could not be split with flax.nnx. Its build_module(config) '
                'helper must return either an NNX-splittable module or '
                '`(module, params_template)` for Flax/custom apply modules.'
            ) from exc
        params_template = serialization.to_state_dict(state)
        module = graphdef

    params = serialization.from_bytes(params_template, params_path.read_bytes())

    return ModelBundle(config=config, params=params, module=module)


def is_multi_device() -> bool:
    """Return ``True`` if more than one global JAX device is available."""
    return len(jax.devices()) > 1


def _none_leaf(value):
    return value is None


def local_device_mesh(devices=None) -> Mesh:
    """Build a one-dimensional mesh over local devices."""
    if devices is None:
        devices = jax.local_devices()
    return Mesh(np.asarray(list(devices), dtype=object), (DEVICE_AXIS_NAME,))


def global_device_mesh(devices=None) -> Mesh:
    """Build a one-dimensional mesh over all global JAX devices."""
    if devices is None:
        devices = jax.devices()
    return Mesh(np.asarray(list(devices), dtype=object), (DEVICE_AXIS_NAME,))


def process_local_sharded_to_global(tree, *, devices=None):
    """Convert process-local device-axis batches into a global sharded array tree."""
    mesh = global_device_mesh(devices)
    sharding = NamedSharding(mesh, DEVICE_AXIS_SPEC)
    global_device_count = int(np.asarray(mesh.devices).size)
    local_device_count = int(jax.local_device_count())

    def _convert(leaf):
        if leaf is None:
            return None
        arr = jnp.asarray(leaf)
        if arr.ndim == 0 or arr.shape[0] != local_device_count:
            raise ValueError(
                'Expected a process-local sharded batch leaf with leading axis '
                f'equal to the local device count ({local_device_count}), got '
                f'shape {arr.shape}.'
            )
        global_shape = (global_device_count,) + tuple(arr.shape[1:])
        return jax.make_array_from_process_local_data(
            sharding,
            arr,
            global_shape=global_shape,
        )

    return jtu.tree_map(_convert, tree, is_leaf=_none_leaf)


def remove_local_device_axis(tree):
    """Remove the size-one local shard axis exposed inside ``jax.shard_map``."""

    def _squeeze(leaf):
        if leaf is None:
            return None
        arr = jnp.asarray(leaf)
        if arr.ndim > 0 and arr.shape[0] == 1:
            return jnp.squeeze(arr, axis=0)
        return arr

    return jtu.tree_map(_squeeze, tree, is_leaf=_none_leaf)


def add_local_device_axis(tree):
    """Add the local shard axis expected by ``jax.shard_map`` outputs."""

    def _expand(leaf):
        if leaf is None:
            return None
        return jnp.expand_dims(jnp.asarray(leaf), axis=0)

    return jtu.tree_map(_expand, tree, is_leaf=_none_leaf)


def shard_map_over_local_devices(
    fn,
    *,
    in_specs,
    out_specs=DEVICE_AXIS_SPEC,
    devices=None,
    check_vma: bool | None = True,
):
    """Compile ``fn`` with ``jax.shard_map`` over the provided device axis."""
    if _shard_map is None:
        raise RuntimeError(
            'JAX multi-device execution requires jax.shard_map. Upgrade JAX or '
            'install a JAX release exposing jax.experimental.shard_map.'
        )
    mesh = local_device_mesh(devices)
    shard_map_kwargs = {
        'mesh': mesh,
        'in_specs': in_specs,
        'out_specs': out_specs,
    }
    shard_map_params = inspect.signature(_shard_map).parameters
    if 'axis_names' in shard_map_params:
        shard_map_kwargs['axis_names'] = {DEVICE_AXIS_NAME}
    if check_vma is not None and 'check_vma' in shard_map_params:
        shard_map_kwargs['check_vma'] = check_vma
    elif check_vma is not None and 'check_rep' in shard_map_params:
        shard_map_kwargs['check_rep'] = check_vma
    mapped = _shard_map(fn, **shard_map_kwargs)
    return jax.jit(mapped)


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
                if key == 'graph_id':
                    pad_vals = -np.ones(pad_shape, dtype=value.dtype)
                else:
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


def _graph_leaf_shapes(graph: jraph.GraphsTuple):
    shape_tree = jtu.tree_map(
        lambda x: None if x is None else np.shape(x),
        graph,
        is_leaf=_none_leaf,
    )
    return jtu.tree_leaves(shape_tree, is_leaf=_none_leaf)


def _mark_padding_graph_ids(
    graph: jraph.GraphsTuple, first_padding_graph: int
) -> jraph.GraphsTuple:
    globals_attr = getattr(graph, 'globals', None)
    if globals_attr is None:
        return graph
    if hasattr(globals_attr, 'items'):
        graph_ids = globals_attr.get('graph_id')
    else:
        graph_ids = getattr(globals_attr, 'graph_id', None)
    if graph_ids is None:
        return graph

    graph_ids = np.asarray(graph_ids)
    if graph_ids.ndim == 0 or graph_ids.shape[0] <= first_padding_graph:
        return graph
    graph_ids = graph_ids.copy()
    graph_ids[first_padding_graph:] = -1

    if hasattr(globals_attr, '_replace'):
        globals_attr = globals_attr._replace(graph_id=graph_ids)
    elif hasattr(globals_attr, 'items'):
        globals_attr = globals_attr.__class__(globals_attr)
        globals_attr['graph_id'] = graph_ids
    else:
        return graph
    return graph._replace(globals=globals_attr)


def _pad_graphs_for_device_stack(
    graphs: list[jraph.GraphsTuple],
    *,
    ensure_padding_graph: bool = False,
) -> list[jraph.GraphsTuple]:
    if len(graphs) <= 1 and not ensure_padding_graph:
        return graphs

    first_shapes = _graph_leaf_shapes(graphs[0])
    if not ensure_padding_graph and all(
        _graph_leaf_shapes(graph) == first_shapes for graph in graphs[1:]
    ):
        return graphs

    max_nodes = max(int(np.asarray(graph.n_node).sum()) for graph in graphs)
    max_edges = max(int(np.asarray(graph.n_edge).sum()) for graph in graphs)
    max_graphs = max(int(np.asarray(graph.n_node).shape[0]) for graph in graphs)

    padded_graphs = []
    for graph in graphs:
        graph_count = int(np.asarray(graph.n_node).shape[0])
        padded = jraph.pad_with_graphs(
            graph,
            n_node=max_nodes + 1,
            n_edge=max_edges + 1,
            n_graph=max_graphs + 1,
        )
        padded_graphs.append(_mark_padding_graph_ids(padded, graph_count))
    return padded_graphs


def shard_graphs_for_devices(
    graph: jraph.GraphsTuple, num_devices: int
) -> list[jraph.GraphsTuple]:
    """Split and pad a graph batch into stackable per-device chunks."""
    return _pad_graphs_for_device_stack(
        split_graphs_for_devices(graph, num_devices),
        ensure_padding_graph=True,
    )


def prepare_sharded_batch(graph, num_devices: int):
    """Prepare a micro-batch for mapped local-device execution."""

    def _ensure_graphs_tuple(item):
        if isinstance(item, jraph.GraphsTuple):
            return item
        if isinstance(item, Sequence) and not isinstance(item, bytes | str):
            return item[0] if len(item) == 1 else jraph.batch_np(item)
        raise TypeError('Expected a GraphsTuple or sequence of GraphsTuples.')

    if isinstance(graph, Sequence) and not isinstance(graph, jraph.GraphsTuple):
        filtered = [g for g in graph if g is not None]
        if len(filtered) != num_devices:
            raise ValueError(
                f'Expected {num_devices} micro-batches for multi-device execution, '
                f'got {len(filtered)}.'
            )
        device_graphs = [_ensure_graphs_tuple(item) for item in filtered]
    else:
        device_graphs = [
            _ensure_graphs_tuple(chunk)
            for chunk in shard_graphs_for_devices(graph, num_devices)
        ]

    device_graphs = _pad_graphs_for_device_stack(device_graphs)
    device_batches = [prepare_single_batch(chunk) for chunk in device_graphs]

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
    'DEVICE_AXIS_NAME',
    'DEVICE_AXIS_SPEC',
    'REPLICATED_SPEC',
    'local_device_mesh',
    'global_device_mesh',
    'process_local_sharded_to_global',
    'remove_local_device_axis',
    'add_local_device_axis',
    'shard_map_over_local_devices',
    'replicate_to_local_devices',
    'unreplicate_from_local_devices',
    'prepare_single_batch',
    'prepare_sharded_batch',
    'split_graphs_for_devices',
    'shard_graphs_for_devices',
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
