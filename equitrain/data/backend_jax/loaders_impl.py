"""Streaming HDF5 data loader for Equitrain datasets.

This module provides fixed-shape batch packing for JAX by reading Equitrain HDF5
shards and padding batches to (n_node, n_edge, n_graph) caps. Graphs are tagged
with a stable global graph_id so prediction outputs can be reordered to match
the original HDF5 order. The loader supports deterministic per-epoch shuffling,
per-process round-robin sharding for distributed runs, and optional
multi-process workers that build batches directly from the HDF5 files.
Workers can be kept alive across epochs (see keep_workers_alive) to avoid
process spawn and repeated HDF5 open costs.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, Full

import jraph
import numpy as np

from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_jax.atoms_to_graphs import AtomsToGraphs
from equitrain.data.configuration import niggli_reduce_inplace
from equitrain.data.format_hdf5 import HDF5Dataset

from .loaders_impl_cache import (
    load_cached_streaming_stats,
    spec_fingerprint,
    stats_payload_from_parts,
    stats_payload_to_parts,
    store_cached_streaming_stats,
)

_RESULT_BATCH = 'batch'
_RESULT_DONE = 'done'
_RESULT_ERROR = 'error'
_INDEX_GRAPH = 'graph'
_INDEX_DONE = 'done'
_INDEX_STOP = 'stop'
_CONVERTER_CACHE: dict[tuple[float, tuple[int, ...]], AtomsToGraphs] = {}


class BatchIteratorWrapper:
    """Wrap an iterator and expose a total_batches_hint attribute."""

    def __init__(self, iterator, total_batches_hint: int):
        """Initialize the wrapper with a source iterator and batch hint."""
        self._iterator = iterator
        self.total_batches_hint = int(total_batches_hint or 0)
        self._lock = threading.Lock()

    def __iter__(self):
        """Return self as an iterator."""
        return self

    def __next__(self):
        """Yield the next batch from the wrapped iterator."""
        with self._lock:
            return next(self._iterator)


@dataclass(frozen=True)
class StreamingDatasetSpec:
    """Configuration describing one HDF5 dataset stream."""

    path: Path
    energy_key: str = 'energy'
    forces_key: str = 'forces'
    stress_key: str = 'stress'
    virials_key: str = 'virials'
    dipole_key: str = 'dipole'
    charges_key: str = 'charges'


@dataclass(frozen=True)
class _StreamingStats:
    """Cached padding caps summary for a single HDF5 dataset."""

    n_nodes: int
    n_edges: int
    n_graphs: int
    n_batches: int | None


def _niggli_reduce_inplace(atoms):
    """Apply Niggli reduction to ASE atoms if periodic and available."""
    try:
        niggli_reduce_inplace(atoms)
    except Exception:  # pragma: no cover - best-effort helper
        return atoms
    return atoms


def _atoms_to_graph(
    *,
    atoms,
    spec: StreamingDatasetSpec,
    cutoff: float,
    z_table: AtomicNumberTable,
    niggli_reduce: bool,
):
    """Convert an ASE atoms object into a jraph GraphsTuple."""
    cache_key = (float(cutoff), tuple(int(z) for z in z_table))
    converter = _CONVERTER_CACHE.get(cache_key)
    if converter is None:
        converter = AtomsToGraphs(
            atomic_numbers=list(z_table),
            r_max=float(cutoff),
            niggli_reduce=False,
        )
        _CONVERTER_CACHE[cache_key] = converter
    if niggli_reduce:
        atoms = atoms.copy()
        _niggli_reduce_inplace(atoms)
    return converter.convert(atoms)


def _has_positive_weight(graph: jraph.GraphsTuple) -> bool:
    """Return True if graph weight is positive or missing."""
    return True


def _with_graph_id(graph: jraph.GraphsTuple, graph_id: int) -> jraph.GraphsTuple:
    """Attach a per-graph identifier to globals for optional reordering."""
    globals_attr = getattr(graph, 'globals', None)
    if globals_attr is None:
        return graph
    graph_id_arr = np.asarray([int(graph_id)], dtype=np.int64)
    if hasattr(globals_attr, '_replace'):
        globals_attr = globals_attr._replace(graph_id=graph_id_arr)
    elif hasattr(globals_attr, 'items'):
        globals_attr = globals_attr.__class__(globals_attr)
        globals_attr['graph_id'] = graph_id_arr
    else:
        return graph
    return graph._replace(globals=globals_attr)


def _mark_padding_graph_ids(
    graph: jraph.GraphsTuple, graph_count: int
) -> jraph.GraphsTuple:
    """Set graph_id=-1 for padded graphs so predictions can filter them out."""
    globals_attr = getattr(graph, 'globals', None)
    if globals_attr is None:
        return graph
    graph_ids = getattr(globals_attr, 'graph_id', None)
    if graph_ids is None:
        return graph
    graph_ids = np.asarray(graph_ids)
    if graph_ids.ndim == 0:
        return graph
    if graph_ids.shape[0] <= int(graph_count):
        return graph
    graph_ids = graph_ids.copy()
    graph_ids[int(graph_count) :] = -1
    if hasattr(globals_attr, '_replace'):
        globals_attr = globals_attr._replace(graph_id=graph_ids)
    elif hasattr(globals_attr, 'items'):
        globals_attr = globals_attr.__class__(globals_attr)
        globals_attr['graph_id'] = graph_ids
    else:
        return graph
    return graph._replace(globals=globals_attr)


def _pack_sizes_by_edge_cap(
    graph_sizes: list[tuple[int, int]],
    edge_cap: int,
) -> list[dict[str, int]]:
    """Greedily pack (nodes, edges) sizes into batches under an edge budget."""
    if not graph_sizes:
        return []

    batches: list[dict[str, int]] = []
    order = sorted(graph_sizes, key=lambda item: item[1], reverse=True)
    for nodes, edges in order:
        placed = False
        for batch in batches:
            if batch['edge_sum'] + edges <= edge_cap:
                batch['edge_sum'] += edges
                batch['node_sum'] += nodes
                batch['graph_count'] += 1
                placed = True
                break
        if not placed:
            batches.append({'edge_sum': edges, 'node_sum': nodes, 'graph_count': 1})
    return batches


def _compute_streaming_stats(
    dataset_path: Path,
    *,
    spec: StreamingDatasetSpec,
    z_table: AtomicNumberTable,
    r_max: float,
    edge_cap: int | None,
    niggli_reduce: bool,
) -> _StreamingStats:
    """Scan a dataset to compute padding caps and batch estimates."""
    dataset = HDF5Dataset(dataset_path, mode='r')
    graph_sizes: list[tuple[int, int]] = []
    max_graph_edges = 0
    try:
        for idx in range(len(dataset)):
            graph = _atoms_to_graph(
                atoms=dataset[idx],
                spec=spec,
                cutoff=r_max,
                z_table=z_table,
                niggli_reduce=niggli_reduce,
            )
            if not _has_positive_weight(graph):
                continue
            g_nodes = int(graph.n_node.sum())
            g_edges = int(graph.n_edge.sum())
            max_graph_edges = max(max_graph_edges, g_edges)
            graph_sizes.append((g_nodes, g_edges))
    finally:
        dataset.close()

    if not graph_sizes:
        raise ValueError(f"No graphs found in '{dataset_path}'.")

    local_edge_cap = edge_cap
    if local_edge_cap is None or local_edge_cap <= 0:
        local_edge_cap = max_graph_edges
    if max_graph_edges > local_edge_cap:
        logging.warning(
            'Requested max edges per batch (%s) is below the largest graph (%s) '
            'in %s. Raising the limit to fit.',
            local_edge_cap,
            max_graph_edges,
            dataset_path.name,
        )
        local_edge_cap = max_graph_edges

    batches = _pack_sizes_by_edge_cap(graph_sizes, int(local_edge_cap))
    max_nodes_per_batch = 0
    max_graphs_per_batch = 0
    for batch in batches:
        max_nodes_per_batch = max(max_nodes_per_batch, batch['node_sum'])
        max_graphs_per_batch = max(max_graphs_per_batch, batch['graph_count'])

    n_nodes = max(max_nodes_per_batch + 1, 2)
    n_graphs = max(max_graphs_per_batch + 1, 2)
    n_batches = len(batches)
    return _StreamingStats(
        n_nodes=n_nodes,
        n_edges=int(local_edge_cap),
        n_graphs=n_graphs,
        n_batches=n_batches,
    )


def _load_or_compute_streaming_stats(
    spec: StreamingDatasetSpec,
    *,
    r_max: float,
    z_table: AtomicNumberTable,
    edge_cap: int | None,
    niggli_reduce: bool,
) -> _StreamingStats:
    """Fetch streaming stats from cache or compute them if missing."""
    dataset_path = Path(spec.path)
    fingerprint = spec_fingerprint(
        spec,
        r_max=r_max,
        atomic_numbers=list(z_table),
        edge_cap=edge_cap,
    )
    cached_payload = load_cached_streaming_stats(dataset_path, fingerprint)
    if cached_payload is not None:
        n_nodes, n_edges, n_graphs, n_batches = stats_payload_to_parts(cached_payload)
        return _StreamingStats(
            n_nodes=n_nodes,
            n_edges=n_edges,
            n_graphs=n_graphs,
            n_batches=n_batches,
        )
    stats = _compute_streaming_stats(
        dataset_path,
        spec=spec,
        z_table=z_table,
        r_max=r_max,
        edge_cap=edge_cap,
        niggli_reduce=niggli_reduce,
    )
    store_cached_streaming_stats(
        dataset_path,
        fingerprint,
        stats_payload_from_parts(
            stats.n_nodes,
            stats.n_edges,
            stats.n_graphs,
            stats.n_batches,
        ),
    )
    return stats


def _graph_worker_main(
    worker_id: int,
    *,
    index_queue,
    result_queue,
    stop_event,
    dataset_specs: Sequence[StreamingDatasetSpec],
    dataset_offsets: Sequence[int],
    dataset_lengths: Sequence[int],
    z_table: AtomicNumberTable,
    r_max: float,
    niggli_reduce: bool,
    n_node: int,
    n_edge: int,
    n_graph: int,
    **_,
):
    """Convert atoms to graphs, pack into batches, and return padded batches."""
    graphs: list[jraph.GraphsTuple] = []
    nodes_sum = 0
    edges_sum = 0
    graph_count = 0
    max_graphs = max(int(n_graph) - 1, 1)
    datasets = [HDF5Dataset(spec.path, mode='r') for spec in dataset_specs]
    total_graphs = int(sum(int(length) for length in dataset_lengths))

    def _result_put(item):
        result_queue.put(item)

    def _reset():
        nonlocal graphs, nodes_sum, edges_sum, graph_count
        graphs = []
        nodes_sum = 0
        edges_sum = 0
        graph_count = 0

    def _flush():
        if not graphs:
            return
        batched = jraph.batch_np(graphs)
        batch = jraph.pad_with_graphs(
            batched,
            n_node=int(n_node),
            n_edge=int(n_edge),
            n_graph=int(n_graph),
        )
        batch = _mark_padding_graph_ids(batch, graph_count)
        _result_put((_RESULT_BATCH, batch, graph_count))
        _reset()

    def _parse_index_message(message):
        if message is None:
            return _INDEX_STOP, None
        if isinstance(message, tuple) and len(message) == 2:
            tag, payload = message
            if tag in (_INDEX_GRAPH, _INDEX_DONE, _INDEX_STOP):
                return tag, payload
        return _INDEX_GRAPH, message

    try:
        if total_graphs <= 0:
            return
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            try:
                message = index_queue.get(timeout=1.0)
            except Empty:
                continue
            tag, payload = _parse_index_message(message)
            if tag == _INDEX_STOP:
                _flush()
                _result_put((_RESULT_DONE, worker_id, None))
                break
            if tag == _INDEX_DONE:
                _flush()
                _result_put((_RESULT_DONE, worker_id, None))
                _reset()
                continue
            graph_id = int(payload)
            if graph_id < 0 or graph_id >= total_graphs:
                continue
            ds_idx = bisect_right(dataset_offsets, graph_id) - 1
            ds_idx = max(ds_idx, 0)
            local_idx = graph_id - int(dataset_offsets[ds_idx])
            if local_idx < 0 or local_idx >= int(dataset_lengths[ds_idx]):
                continue
            atoms = datasets[ds_idx][int(local_idx)]
            spec = dataset_specs[int(ds_idx)]
            graph = _atoms_to_graph(
                atoms=atoms,
                spec=spec,
                cutoff=r_max,
                z_table=z_table,
                niggli_reduce=niggli_reduce,
            )
            if not _has_positive_weight(graph):
                continue
            graph = _with_graph_id(graph, graph_id)
            nodes = int(graph.n_node.sum())
            edges = int(graph.n_edge.sum())
            if nodes >= int(n_node) or edges > int(n_edge):
                _result_put(
                    (
                        _RESULT_ERROR,
                        worker_id,
                        f'Graph exceeds padding limits (nodes={nodes} edges={edges}, '
                        f'caps n_node={n_node} n_edge={n_edge}).',
                    )
                )
                if stop_event is not None:
                    stop_event.set()
                break
            if graphs and (
                nodes_sum + nodes >= int(n_node)
                or edges_sum + edges > int(n_edge)
                or graph_count >= max_graphs
            ):
                _flush()
            graphs.append(graph)
            graph_count += 1
            nodes_sum += nodes
            edges_sum += edges
            if graph_count >= max_graphs:
                _flush()
    except Exception as exc:  # pragma: no cover - worker crashes are unexpected
        _result_put((_RESULT_ERROR, worker_id, repr(exc)))
        if stop_event is not None:
            stop_event.set()
    finally:
        for dataset in datasets:
            dataset.close()
        if graphs:
            _flush()
        _result_put((_RESULT_DONE, worker_id, None))


class StreamingGraphDataLoader:
    """Stream HDF5-backed graphs with fixed-size padded batches.

    When keep_workers_alive is enabled, a persistent worker pool is reused
    across epochs to avoid process respawn and repeated HDF5 open costs.
    """

    def __init__(
        self,
        *,
        datasets: Sequence[HDF5Dataset],
        dataset_specs: Sequence[StreamingDatasetSpec] | None = None,
        z_table: AtomicNumberTable,
        r_max: float,
        n_node: int | None,
        n_edge: int | None,
        shuffle: bool = False,
        seed: int | None = None,
        niggli_reduce: bool = False,
        max_batches: int | None = None,
        prefetch_batches: int | None = None,
        num_workers: int | None = None,
        pad_graphs: int | None = None,
        graph_multiple: int | None = None,
        keep_workers_alive: bool = True,
    ):
        if not datasets:
            raise ValueError('Expected at least one dataset.')
        self._datasets = list(datasets)
        base_paths: list[Path] = []
        for ds in self._datasets:
            ds_path = getattr(ds, '_filename', getattr(ds, 'filename', None))
            if ds_path is None:
                raise ValueError('Unable to determine dataset path for worker loading.')
            base_paths.append(Path(ds_path))
        if dataset_specs is None:
            dataset_specs = [StreamingDatasetSpec(path=path) for path in base_paths]
        if len(dataset_specs) != len(self._datasets):
            raise ValueError(
                'dataset_specs must match datasets length '
                f'({len(dataset_specs)} vs {len(self._datasets)}).'
            )
        normalized_specs: list[StreamingDatasetSpec] = []
        for spec in dataset_specs:
            if isinstance(spec.path, Path):
                normalized_specs.append(spec)
            else:
                normalized_specs.append(replace(spec, path=Path(spec.path)))
        self._dataset_specs = normalized_specs
        self._dataset_paths = [Path(spec.path) for spec in self._dataset_specs]
        self._z_table = z_table
        self._cutoff = float(r_max)
        self._shuffle = bool(shuffle)
        self._seed = None if seed is None else int(seed)
        self._niggli_reduce = bool(niggli_reduce)
        self._max_batches = max_batches
        self._keep_workers_alive = bool(keep_workers_alive)
        prefetched = prefetch_batches
        worker_count = int(num_workers or 0)
        if prefetched is None:
            prefetched = 10 * max(worker_count, 1)
        self._prefetch_batches = max(int(prefetched or 0), 0)
        self._num_workers = max(worker_count, 0)
        self._graph_multiple = max(int(graph_multiple or 1), 1)
        self._mp_ctx = None
        self._index_queue = None
        self._result_queue = None
        self._stop_event = None
        self._worker_procs: list[mp.Process] = []
        self._worker_pool_started = False

        self._dataset_offsets: list[int] = []
        self._dataset_lengths: list[int] = []
        offset = 0
        for dataset in self._datasets:
            length = len(dataset)
            self._dataset_offsets.append(offset)
            self._dataset_lengths.append(length)
            offset += length

        self._dataset_estimated_batches: list[int | None] | None = None
        if n_node is None or n_edge is None or pad_graphs is None:
            stats_list = [
                _load_or_compute_streaming_stats(
                    spec,
                    r_max=self._cutoff,
                    z_table=self._z_table,
                    edge_cap=n_edge,
                    niggli_reduce=self._niggli_reduce,
                )
                for spec in self._dataset_specs
            ]
            if stats_list:
                self._dataset_estimated_batches = [
                    stats.n_batches for stats in stats_list
                ]
                stats_n_nodes = max(int(stats.n_nodes) for stats in stats_list)
                stats_n_edges = max(int(stats.n_edges) for stats in stats_list)
                stats_n_graphs = max(int(stats.n_graphs) for stats in stats_list)
                if n_node is None:
                    n_node = stats_n_nodes
                else:
                    n_node = max(int(n_node), stats_n_nodes)
                if n_edge is None:
                    n_edge = stats_n_edges
                else:
                    n_edge = max(int(n_edge), stats_n_edges)
                if pad_graphs is None:
                    pad_graphs = stats_n_graphs
                else:
                    pad_graphs = max(int(pad_graphs), stats_n_graphs)
                self.estimated_batches = sum(
                    int(estimate)
                    for estimate in self._dataset_estimated_batches
                    if estimate
                )

        if n_node is None or n_edge is None or pad_graphs is None:
            raise ValueError('Failed to determine n_node, n_edge, and n_graph limits.')

        if self._graph_multiple > 1:
            pad_graphs = int(
                ((int(pad_graphs) + self._graph_multiple - 1) // self._graph_multiple)
                * self._graph_multiple
            )

        self._n_node = int(max(n_node, 1))
        self._n_edge = int(max(n_edge, 1))
        self._n_graph = int(max(pad_graphs, 2))
        self._last_padding_summary: dict[str, int] | None = None

        for dataset in self._datasets:
            dataset.close()
        self._datasets = []

        self.graphs = getattr(self, 'graphs', None)
        self.streaming = True
        self.total_graphs = int(sum(self._dataset_lengths))
        self.total_nodes = getattr(self, 'total_nodes', None)
        self.total_edges = getattr(self, 'total_edges', None)

    def _graph_ids_for_epoch(
        self,
        *,
        epoch: int,
        seed: int | None,
        process_count: int,
        process_index: int,
    ) -> Sequence[int]:
        total_graphs = int(sum(self._dataset_lengths))
        if total_graphs <= 0:
            return ()
        if not self._shuffle:
            return range(process_index, total_graphs, process_count)
        seed_value = seed
        if seed_value is None:
            seed_value = self._seed or 0
        seed_value = int(seed_value) + int(epoch)
        rng = np.random.default_rng(seed_value)
        indices = np.arange(total_graphs, dtype=np.int64)
        rng.shuffle(indices)
        if process_count > 1:
            indices = indices[process_index::process_count]
        return indices

    def _iter_single_process(
        self,
        *,
        graph_ids: Sequence[int],
    ) -> Iterator[tuple[jraph.GraphsTuple, int]]:
        """Yield batches and graph counts without multiprocessing."""
        max_graphs = max(int(self._n_graph) - 1, 1)
        graphs: list[jraph.GraphsTuple] = []
        nodes_sum = 0
        edges_sum = 0
        graph_count = 0

        def _flush():
            nonlocal graphs, nodes_sum, edges_sum, graph_count
            if not graphs:
                return None
            batched = jraph.batch_np(graphs)
            batch = jraph.pad_with_graphs(
                batched,
                n_node=self._n_node,
                n_edge=self._n_edge,
                n_graph=self._n_graph,
            )
            batch = _mark_padding_graph_ids(batch, graph_count)
            result = (batch, graph_count)
            graphs = []
            nodes_sum = 0
            edges_sum = 0
            graph_count = 0
            return result

        datasets = [HDF5Dataset(spec.path, mode='r') for spec in self._dataset_specs]
        try:
            for graph_id in graph_ids:
                ds_idx = bisect_right(self._dataset_offsets, int(graph_id)) - 1
                ds_idx = max(ds_idx, 0)
                local_idx = int(graph_id) - int(self._dataset_offsets[ds_idx])
                if local_idx < 0 or local_idx >= int(self._dataset_lengths[ds_idx]):
                    continue
                atoms = datasets[ds_idx][int(local_idx)]
                spec = self._dataset_specs[ds_idx]
                graph = _atoms_to_graph(
                    atoms=atoms,
                    spec=spec,
                    cutoff=self._cutoff,
                    z_table=self._z_table,
                    niggli_reduce=self._niggli_reduce,
                )
                if not _has_positive_weight(graph):
                    continue
                graph = _with_graph_id(graph, int(graph_id))
                nodes = int(graph.n_node.sum())
                edges = int(graph.n_edge.sum())
                if nodes >= self._n_node or edges > self._n_edge:
                    raise ValueError(
                        'Graph exceeds padding limits '
                        f'(nodes={nodes} edges={edges}, '
                        f'caps n_node={self._n_node} n_edge={self._n_edge}).'
                    )
                if graphs and (
                    nodes_sum + nodes >= self._n_node
                    or edges_sum + edges > self._n_edge
                    or graph_count >= max_graphs
                ):
                    flushed = _flush()
                    if flushed is not None:
                        yield flushed
                graphs.append(graph)
                graph_count += 1
                nodes_sum += nodes
                edges_sum += edges
                if graph_count >= max_graphs:
                    flushed = _flush()
                    if flushed is not None:
                        yield flushed
        finally:
            for dataset in datasets:
                dataset.close()
        flushed = _flush()
        if flushed is not None:
            yield flushed

    def _ensure_worker_pool(self) -> None:
        """Start a persistent worker pool if configured."""
        if self._worker_pool_started or self._num_workers <= 0:
            return
        ctx = mp.get_context('spawn')
        self._mp_ctx = ctx
        worker_count = max(self._num_workers, 1)
        index_queue_capacity = max(worker_count * 64, 1)
        result_queue_capacity = max(worker_count * 32, 1)
        self._index_queue = ctx.Queue(index_queue_capacity)
        self._result_queue = ctx.Queue(result_queue_capacity)
        self._stop_event = ctx.Event()
        self._worker_procs = []
        for worker_id in range(worker_count):
            proc = ctx.Process(
                target=_graph_worker_main,
                args=(worker_id,),
                kwargs={
                    'index_queue': self._index_queue,
                    'result_queue': self._result_queue,
                    'stop_event': self._stop_event,
                    'dataset_specs': self._dataset_specs,
                    'dataset_offsets': self._dataset_offsets,
                    'dataset_lengths': self._dataset_lengths,
                    'z_table': self._z_table,
                    'r_max': self._cutoff,
                    'niggli_reduce': self._niggli_reduce,
                    'n_node': self._n_node,
                    'n_edge': self._n_edge,
                    'n_graph': self._n_graph,
                },
            )
            proc.daemon = True
            proc.start()
            self._worker_procs.append(proc)
        self._worker_pool_started = True

    def _shutdown_worker_pool(self) -> None:
        """Stop any persistent workers and clean up queues."""
        if not self._worker_pool_started:
            return
        if self._stop_event is not None:
            self._stop_event.set()
        if self._index_queue is not None:
            for _ in range(max(self._num_workers, 1)):
                try:
                    self._index_queue.put((_INDEX_STOP, None), timeout=1.0)
                except Full:
                    pass
        for proc in self._worker_procs:
            proc.join(timeout=1.0)
        if self._index_queue is not None:
            self._index_queue.close()
        if self._result_queue is not None:
            self._result_queue.close()
        self._mp_ctx = None
        self._index_queue = None
        self._result_queue = None
        self._stop_event = None
        self._worker_procs = []
        self._worker_pool_started = False

    def _iter_multi_process(
        self,
        *,
        graph_ids: Sequence[int],
    ) -> Iterator[tuple[jraph.GraphsTuple, int]]:
        """Yield batches from worker processes that read HDF5 directly."""
        worker_count = max(self._num_workers, 1)
        if self._keep_workers_alive:
            self._ensure_worker_pool()
            index_queue = self._index_queue
            result_queue = self._result_queue
            stop_event = self._stop_event
            worker_procs = self._worker_procs
        else:
            ctx = mp.get_context('spawn')
            index_queue_capacity = max(worker_count * 64, 1)
            index_queue = ctx.Queue(index_queue_capacity)
            result_queue_capacity = max(worker_count * 32, 1)
            result_queue = ctx.Queue(result_queue_capacity)
            stop_event = ctx.Event()

            worker_procs = []
            for worker_id in range(worker_count):
                proc = ctx.Process(
                    target=_graph_worker_main,
                    args=(worker_id,),
                    kwargs={
                        'index_queue': index_queue,
                        'result_queue': result_queue,
                        'stop_event': stop_event,
                        'dataset_specs': self._dataset_specs,
                        'dataset_offsets': self._dataset_offsets,
                        'dataset_lengths': self._dataset_lengths,
                        'z_table': self._z_table,
                        'r_max': self._cutoff,
                        'niggli_reduce': self._niggli_reduce,
                        'n_node': self._n_node,
                        'n_edge': self._n_edge,
                        'n_graph': self._n_graph,
                    },
                )
                proc.daemon = True
                proc.start()
                worker_procs.append(proc)

        def _index_feeder():
            for graph_id in graph_ids:
                while True:
                    if stop_event is not None and stop_event.is_set():
                        return
                    try:
                        index_queue.put((_INDEX_GRAPH, int(graph_id)), timeout=1.0)
                        break
                    except Full:
                        continue
            for _ in range(worker_count):
                while True:
                    if stop_event is not None and stop_event.is_set():
                        return
                    try:
                        index_queue.put((_INDEX_DONE, None), timeout=1.0)
                        break
                    except Full:
                        continue

        feeder = threading.Thread(target=_index_feeder, daemon=True)
        feeder.start()

        def _check_worker_health():
            for idx, proc in enumerate(worker_procs):
                if proc.is_alive():
                    continue
                exit_code = proc.exitcode
                if exit_code is None:
                    continue
                if stop_event is not None:
                    stop_event.set()
                if self._keep_workers_alive:
                    self._shutdown_worker_pool()
                raise RuntimeError(
                    f'Graph worker {idx} exited unexpectedly with code {exit_code}.'
                )

        finished_workers = 0
        try:
            while finished_workers < worker_count:
                try:
                    tag, payload_a, payload_b = result_queue.get(timeout=1.0)
                except Empty:
                    if stop_event is not None and stop_event.is_set():
                        break
                    _check_worker_health()
                    continue
                if tag == _RESULT_DONE:
                    finished_workers += 1
                    continue
                if tag == _RESULT_ERROR:
                    if stop_event is not None:
                        stop_event.set()
                    if self._keep_workers_alive:
                        self._shutdown_worker_pool()
                    raise RuntimeError(f'Graph worker {payload_a} failed: {payload_b}')
                if tag == _RESULT_BATCH:
                    yield payload_a, payload_b
        finally:
            if self._keep_workers_alive and finished_workers < worker_count:
                if stop_event is not None:
                    stop_event.set()
                self._shutdown_worker_pool()
            if not self._keep_workers_alive and stop_event is not None:
                stop_event.set()
            if not self._keep_workers_alive:
                for proc in worker_procs:
                    proc.join(timeout=1)
            feeder.join(timeout=1)

    def iter_batches(
        self,
        *,
        epoch: int,
        seed: int | None,
        process_count: int,
        process_index: int,
    ) -> Iterator[jraph.GraphsTuple]:
        """Yield batches for the given epoch and process shard."""
        if process_count <= 0:
            raise ValueError('process_count must be a positive integer.')
        if process_index < 0 or process_index >= process_count:
            raise ValueError(
                f'process_index {process_index} is out of range for process_count '
                f'{process_count}.'
            )
        graph_ids = self._graph_ids_for_epoch(
            epoch=epoch,
            seed=seed,
            process_count=process_count,
            process_index=process_index,
        )

        total_batches_hint = 0
        if self._dataset_estimated_batches:
            total_batches_hint = sum(
                int(estimate)
                for estimate in self._dataset_estimated_batches
                if estimate
            )
        if not total_batches_hint:
            total_graphs = int(sum(self._dataset_lengths))
            max_graphs = max(int(self._n_graph) - 1, 1)
            total_batches_hint = int(np.ceil(float(total_graphs) / float(max_graphs)))
        if process_count > 1 and total_batches_hint:
            total_batches_hint = int(
                np.ceil(float(total_batches_hint) / float(process_count))
            )
        if self._max_batches is not None:
            total_batches_hint = min(total_batches_hint, int(self._max_batches))
        total_batches_hint = max(total_batches_hint, 0)

        if self._num_workers > 0:
            source_iter = self._iter_multi_process(
                graph_ids=graph_ids,
            )
        else:
            source_iter = self._iter_single_process(
                graph_ids=graph_ids,
            )

        def _iter():
            graphs_count = 0
            nodes_count = 0
            edges_count = 0
            batches_count = 0
            produced = 0
            try:
                for batch, graph_count in source_iter:
                    if self._max_batches is not None and produced >= self._max_batches:
                        break
                    produced += 1
                    graph_total = int(graph_count)
                    graphs_count += graph_total
                    if graph_total > 0:
                        nodes_count += int(np.sum(batch.n_node[:graph_total]))
                        edges_count += int(np.sum(batch.n_edge[:graph_total]))
                    batches_count += 1
                    yield batch
            finally:
                if hasattr(source_iter, 'close'):
                    source_iter.close()
                if batches_count:
                    padded_nodes = int(self._n_node) * batches_count
                    padded_edges = int(self._n_edge) * batches_count
                    padded_graphs = int(self._n_graph) * batches_count
                    pad_nodes = max(padded_nodes - nodes_count, 0)
                    pad_edges = max(padded_edges - edges_count, 0)
                    pad_graphs = max(padded_graphs - graphs_count, 0)
                    self._last_padding_summary = {
                        'batches': batches_count,
                        'pad_nodes': pad_nodes,
                        'pad_edges': pad_edges,
                        'pad_graphs': pad_graphs,
                        'padded_nodes': padded_nodes,
                        'padded_edges': padded_edges,
                        'padded_graphs': padded_graphs,
                    }

        return BatchIteratorWrapper(_iter(), total_batches_hint)

    def __iter__(self):
        """Iterate over batches for a single-process epoch."""
        iterator = self.iter_batches(
            epoch=0,
            seed=self._seed,
            process_count=1,
            process_index=0,
        )
        yield from iterator

    def __len__(self):
        """Return the number of batches for the current packing configuration."""
        return self.approx_length()

    def approx_length(self) -> int:
        """Estimate number of batches without forcing a prepass."""
        total_graphs = getattr(self, 'total_graphs', None)
        total_nodes = getattr(self, 'total_nodes', None)
        total_edges = getattr(self, 'total_edges', None)
        estimated_batches = getattr(self, 'estimated_batches', None)

        estimates: list[int] = []
        if estimated_batches is not None:
            estimates.append(int(estimated_batches))
        max_graphs = max(int(self._n_graph) - 1, 1)
        if total_graphs is not None:
            estimates.append(int(np.ceil(float(total_graphs) / float(max_graphs))))
        if total_nodes is not None:
            max_nodes = max(int(self._n_node) - 1, 1)
            estimates.append(int(np.ceil(float(total_nodes) / float(max_nodes))))
        if total_edges is not None:
            max_edges = max(int(self._n_edge), 1)
            estimates.append(int(np.ceil(float(total_edges) / float(max_edges))))

        if estimates:
            approx = max(estimates)
            if self._max_batches is not None:
                approx = min(approx, int(self._max_batches))
            return max(1, approx)
        return 1

    def close(self) -> None:
        """Close datasets and release cached resources."""
        self._shutdown_worker_pool()
        for dataset in self._datasets:
            dataset.close()
        self._datasets = []


def get_hdf5_dataloader(
    *,
    data_file: Path | str | Sequence[Path | str],
    atomic_numbers: AtomicNumberTable,
    r_max: float,
    shuffle: bool,
    max_nodes: int | None,
    max_edges: int | None,
    seed: int | None = None,
    niggli_reduce: bool = False,
    max_batches: int | None = None,
    prefetch_batches: int | None = None,
    num_workers: int | None = None,
    dataset_specs: Sequence[StreamingDatasetSpec] | None = None,
    graph_multiple: int | None = None,
    keep_workers_alive: bool = True,
) -> StreamingGraphDataLoader:
    """Create a StreamingGraphDataLoader from one or more HDF5 files."""
    if data_file is None:
        raise ValueError('data_file must be provided.')
    if isinstance(data_file, (list, tuple)):
        paths = [Path(path) for path in data_file]
    else:
        paths = [Path(data_file)]
    datasets = [HDF5Dataset(path, mode='r') for path in paths]
    if dataset_specs is None:
        dataset_specs = [StreamingDatasetSpec(path=path) for path in paths]
    return StreamingGraphDataLoader(
        datasets=datasets,
        dataset_specs=dataset_specs,
        z_table=atomic_numbers,
        r_max=r_max,
        n_node=max_nodes,
        n_edge=max_edges,
        shuffle=shuffle,
        seed=seed,
        niggli_reduce=niggli_reduce,
        max_batches=max_batches,
        prefetch_batches=prefetch_batches,
        num_workers=num_workers,
        graph_multiple=graph_multiple,
        keep_workers_alive=keep_workers_alive,
    )


GraphDataLoader = StreamingGraphDataLoader


__all__ = [
    'StreamingDatasetSpec',
    'StreamingGraphDataLoader',
    'GraphDataLoader',
    'get_hdf5_dataloader',
]
