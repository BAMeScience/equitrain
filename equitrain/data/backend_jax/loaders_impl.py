from __future__ import annotations

import threading
from collections.abc import Iterable
from pathlib import Path
from queue import Queue

import jraph
import numpy as np
import multiprocessing as mp

from equitrain.data.backend_jax.atoms_to_graphs import (
    AtomsToGraphs,
    graph_from_configuration,
)
from equitrain.data.configuration import Configuration, niggli_reduce_inplace
from equitrain.data.format_hdf5.dataset import HDF5Dataset


_RESULT_DATA = 'data'
_RESULT_DONE = 'done'
_RESULT_ERROR = 'error'


def _graph_worker_main(
    worker_id: int,
    dataset_paths: list[Path],
    cutoff: float,
    z_table,
    niggli_reduce: bool,
    task_queue,
    result_queue,
    stop_event,
):
    """Process worker fetching HDF5 structures and converting to graphs."""
    local_datasets: dict[int, HDF5Dataset] = {}
    try:
        while not stop_event.is_set():
            task = task_queue.get()
            if task is None:
                break
            seq_id, ds_idx, sample_idx = task
            dataset = local_datasets.get(ds_idx)
            if dataset is None:
                dataset = HDF5Dataset(dataset_paths[ds_idx], mode='r')
                local_datasets[ds_idx] = dataset
            atoms = dataset[sample_idx]
            if niggli_reduce:
                atoms = atoms.copy()
                niggli_reduce_inplace(atoms)
            conf = Configuration.from_atoms(atoms)
            graph = graph_from_configuration(conf, cutoff=cutoff, z_table=z_table)
            result_queue.put((_RESULT_DATA, seq_id, graph))
    except Exception as exc:  # pragma: no cover - worker crashes are unexpected
        stop_event.set()
        result_queue.put((_RESULT_ERROR, worker_id, repr(exc)))
    finally:
        for dataset in local_datasets.values():
            dataset.close()
        result_queue.put((_RESULT_DONE, worker_id, None))


class GraphDataLoader:
    """Streaming graph data loader compatible with JAX/Flax training loops."""

    def __init__(
        self,
        *,
        datasets: list[HDF5Dataset],
        z_table=None,
        r_max: float | None = None,
        n_node: int | None,
        n_edge: int | None,
        n_graph: int | None,
        max_batches: int | None = None,
        shuffle: bool = False,
        seed: int | None = None,
        niggli_reduce: bool = False,
        prefetch_batches: int | None = None,
        num_workers: int | None = None,
        graph_multiple: int | None = None,
    ) -> None:
        """
        h5_sources: list of HDF5Dataset instances or HDF5 file paths to stream.
        """
        if not datasets:
            raise ValueError('At least one HDF5 dataset must be provided.')
        if z_table is None or r_max is None:
            raise ValueError('z_table and r_max required when streaming from HDF5.')

        self._n_graph = None if n_graph is None else max(int(n_graph), 1)
        self._max_batches = max_batches
        self._shuffle = shuffle
        self._seed = seed
        self._niggli_reduce = bool(niggli_reduce)
        self._prefetch_batches = int(prefetch_batches or 0)
        self._num_workers = max(int(num_workers or 0), 0)
        self._graph_multiple = max(int(graph_multiple or 1), 1)
        self._pack_info: dict | None = None

        self._datasets = list(datasets)
        if n_node is None or n_edge is None:
            est_nodes, est_edges = _estimate_caps(self._datasets, z_table, r_max)
            if n_node is None:
                n_node = est_nodes
            if n_edge is None:
                n_edge = est_edges

        self._n_node = int(max(n_node or 0, 1))
        self._n_edge = int(max(n_edge or 0, 1))
        self._cutoff = float(r_max)
        self._z_table = z_table
        self._dataset_paths: list[Path] = []
        for ds in self._datasets:
            ds_path = getattr(ds, '_filename', getattr(ds, 'filename', None))
            if ds_path is None:
                raise ValueError('Unable to determine dataset path for worker loading.')
            self._dataset_paths.append(Path(ds_path))

        self._graph_iter_fn = self._graph_iterator
        self._pending_info: dict | None = None

    def _pack(self):
        batches, info = pack_graphs_greedy(
            graph_iter_fn=self._graph_iter_fn,
            max_edges_per_batch=self._n_edge,
            max_nodes_per_batch=self._n_node,
            batch_size_limit=self._n_graph,
            graph_multiple=self._graph_multiple,
        )
        self._pack_info = info
        return batches, info

    def _target_graphs(self):
        if self._max_batches is None or self._n_graph is None:
            return None
        return self._max_batches * self._n_graph

    def _convert_atoms_to_graph(self, atoms):
        if self._niggli_reduce:
            atoms = atoms.copy()
            niggli_reduce_inplace(atoms)
        conf = Configuration.from_atoms(atoms)
        return graph_from_configuration(
            conf, cutoff=self._cutoff, z_table=self._z_table
        )

    def _task_iterator(self):
        rng = np.random.default_rng(self._seed)
        target = self._target_graphs()
        emitted = 0
        dataset_indices = list(range(len(self._datasets)))
        if self._shuffle and len(dataset_indices) > 1:
            rng.shuffle(dataset_indices)
        for ds_idx in dataset_indices:
            ds = self._datasets[ds_idx]
            indices = np.arange(len(ds))
            if self._shuffle and len(indices) > 1:
                rng.shuffle(indices)
            for idx in indices:
                seq_id = emitted
                emitted += 1
                yield (seq_id, ds_idx, int(idx))
                if target is not None and emitted >= target:
                    return

    def _iter_graphs_single(self):
        rng = np.random.default_rng(self._seed)
        target = self._target_graphs()
        emitted = 0
        sources = list(self._datasets)
        if self._shuffle and len(sources) > 1:
            rng.shuffle(sources)
        for ds in sources:
            indices = np.arange(len(ds))
            if self._shuffle and len(indices) > 1:
                rng.shuffle(indices)
            for idx in indices:
                atoms = ds[idx]
                graph = self._convert_atoms_to_graph(atoms)
                yield graph
                emitted += 1
                if target is not None and emitted >= target:
                    return

    def _iter_graphs_parallel(self):
        worker_count = max(self._num_workers, 1)
        ctx = mp.get_context('spawn')
        task_queue = ctx.Queue(max(worker_count * 4, 1))
        result_queue = ctx.Queue(max(worker_count * 4, 1))
        stop_event = ctx.Event()

        def _task_producer():
            try:
                for task in self._task_iterator():
                    if stop_event.is_set():
                        break
                    task_queue.put(task)
            finally:
                for _ in range(worker_count):
                    task_queue.put(None)

        producer = threading.Thread(target=_task_producer, daemon=True)
        producer.start()

        processes = []
        for worker_idx in range(worker_count):
            proc = ctx.Process(
                target=_graph_worker_main,
                args=(
                    worker_idx,
                    self._dataset_paths,
                    self._cutoff,
                    self._z_table,
                    self._niggli_reduce,
                    task_queue,
                    result_queue,
                    stop_event,
                ),
            )
            proc.daemon = True
            proc.start()
            processes.append(proc)

        finished_workers = 0
        next_seq = 0
        pending: dict[int, jraph.GraphsTuple] = {}
        try:
            while finished_workers < worker_count:
                tag, payload_a, payload_b = result_queue.get()
                if tag == _RESULT_DONE:
                    finished_workers += 1
                    continue
                if tag == _RESULT_ERROR:
                    stop_event.set()
                    for proc in processes:
                        proc.join(timeout=1)
                    raise RuntimeError(
                        f'Graph worker {payload_a} failed: {payload_b}'
                    )
                seq_id = payload_a
                graph = payload_b
                if seq_id == next_seq:
                    yield graph
                    next_seq += 1
                    while next_seq in pending:
                        yield pending.pop(next_seq)
                        next_seq += 1
                else:
                    pending[seq_id] = graph
        finally:
            stop_event.set()
            producer.join(timeout=1)
            for proc in processes:
                proc.join()

    def _graph_iterator(self):
        if self._num_workers <= 1:
            return self._iter_graphs_single()
        return self._iter_graphs_parallel()

    def __iter__(self):
        batches, _ = self._pack()
        if self._prefetch_batches > 0:
            queue: Queue = Queue(maxsize=self._prefetch_batches)
            sentinel = object()

            def _producer():
                try:
                    for item in batches:
                        queue.put(item)
                finally:
                    queue.put(sentinel)

            threading.Thread(target=_producer, daemon=True).start()

            while True:
                item = queue.get()
                if item is sentinel:
                    break
                yield item
        else:
            yield from batches

    def __len__(self):
        if self._pack_info and self._pack_info.get('total_batches') is not None:
            return int(self._pack_info['total_batches'])
        _, info = self._pack()
        return int(info.get('total_batches', 0))

    def pack_info(self) -> dict:
        """Return metadata from the most recent packing run."""
        if not self._pack_info:
            _, info = self._pack()
            return dict(info)
        return dict(self._pack_info)


def pack_graphs_greedy(
    *,
    graph_iter_fn: callable,
    max_edges_per_batch: int,
    max_nodes_per_batch: int | None = None,
    batch_size_limit: int | None = None,
    graph_multiple: int = 1,
) -> tuple[Iterable[jraph.GraphsTuple], dict]:
    """
    Greedily pack graphs into batches limited by edge/node totals.

    The packing is two-pass to keep XLA shapes stable:
    1) Scan to determine padding targets (graphs/nodes/edges) and count batches.
    2) Yield padded batches with fixed shapes. Graphs exceeding the per-graph
       caps are dropped.
    """

    def _make_iter():
        return iter(graph_iter_fn())

    def _scan():
        current_len = edge_sum = node_sum = 0
        max_graphs = max_total_nodes = max_total_edges = 0
        dropped = batches = 0
        graphs_seen = kept = 0
        max_single_nodes = max_single_edges = 0
        for g in _make_iter():
            g_edges = int(g.n_edge.sum())
            g_nodes = int(g.n_node.sum())
            graphs_seen += 1
            max_single_nodes = max(max_single_nodes, g_nodes)
            max_single_edges = max(max_single_edges, g_edges)
            if g_edges > max_edges_per_batch or (
                max_nodes_per_batch is not None and g_nodes > max_nodes_per_batch
            ):
                dropped += 1
                continue
            kept += 1
            would_edges = edge_sum + g_edges
            would_nodes = node_sum + g_nodes
            if current_len and (
                would_edges > max_edges_per_batch
                or (
                    max_nodes_per_batch is not None
                    and would_nodes > max_nodes_per_batch
                )
                or (batch_size_limit is not None and current_len >= batch_size_limit)
            ):
                max_graphs = max(max_graphs, current_len)
                max_total_nodes = max(max_total_nodes, node_sum)
                max_total_edges = max(max_total_edges, edge_sum)
                batches += 1
                current_len = edge_sum = node_sum = 0
            current_len += 1
            edge_sum += g_edges
            node_sum += g_nodes
        if current_len:
            max_graphs = max(max_graphs, current_len)
            max_total_nodes = max(max_total_nodes, node_sum)
            max_total_edges = max(max_total_edges, edge_sum)
            batches += 1
        return (
            max_graphs + 1 if max_graphs else 1,
            max_total_nodes + 1 if max_total_nodes else 1,
            max_total_edges + 1 if max_total_edges else 1,
            dropped,
            batches,
            graphs_seen,
            kept,
            max_single_nodes,
            max_single_edges,
        )

    (
        pad_graphs,
        pad_nodes,
        pad_edges,
        dropped,
        total_batches,
        graphs_seen,
        kept,
        max_single_nodes,
        max_single_edges,
    ) = _scan()

    graph_multiple = max(int(graph_multiple or 1), 1)
    if graph_multiple > 1:
        pad_graphs = ((pad_graphs + graph_multiple - 1) // graph_multiple) * graph_multiple

    def _empty_graph_like(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        def _zero_nodes(arr):
            shape = (0,) + arr.shape[1:]
            return np.zeros(shape, dtype=arr.dtype)

        nodes = graph.nodes.__class__()
        for key, value in graph.nodes.items():
            nodes[key] = _zero_nodes(value)

        edges = graph.edges.__class__()
        for key, value in graph.edges.items():
            edges[key] = _zero_nodes(value)

        globals_dict = graph.globals.__class__()
        for key, value in graph.globals.items():
            globals_dict[key] = np.zeros_like(value)

        return jraph.GraphsTuple(
            nodes=nodes,
            edges=edges,
            senders=np.zeros((0,), dtype=graph.senders.dtype),
            receivers=np.zeros((0,), dtype=graph.receivers.dtype),
            globals=globals_dict,
            n_node=np.asarray([0], dtype=np.int32),
            n_edge=np.asarray([0], dtype=np.int32),
        )

    def _pad_graph_list(graphs: list[jraph.GraphsTuple]):
        if graph_multiple <= 1 or not graphs:
            return
        remainder = len(graphs) % graph_multiple
        if remainder == 0:
            return
        template = graphs[0]
        dummy = _empty_graph_like(template)
        for _ in range(graph_multiple - remainder):
            graphs.append(dummy)

    def _iter():
        current: list[jraph.GraphsTuple] = []
        edge_sum = node_sum = 0
        for g in _make_iter():
            g_edges = int(g.n_edge.sum())
            g_nodes = int(g.n_node.sum())
            if g_edges > max_edges_per_batch or (
                max_nodes_per_batch is not None and g_nodes > max_nodes_per_batch
            ):
                continue
            would_edges = edge_sum + g_edges
            would_nodes = node_sum + g_nodes
            if current and (
                would_edges > max_edges_per_batch
                or (
                    max_nodes_per_batch is not None
                    and would_nodes > max_nodes_per_batch
                )
                or (batch_size_limit is not None and len(current) >= batch_size_limit)
            ):
                _pad_graph_list(current)
                batched = jraph.batch_np(current)
                yield jraph.pad_with_graphs(
                    batched,
                    n_node=pad_nodes,
                    n_edge=pad_edges,
                    n_graph=pad_graphs,
                )
                current = []
                edge_sum = node_sum = 0
            current.append(g)
            edge_sum += g_edges
            node_sum += g_nodes
        if current:
            _pad_graph_list(current)
            batched = jraph.batch_np(current)
            yield jraph.pad_with_graphs(
                batched,
                n_node=pad_nodes,
                n_edge=pad_edges,
                n_graph=pad_graphs,
            )

    info = dict(
        dropped=dropped,
        total_batches=total_batches,
        pad_graphs=pad_graphs,
        pad_nodes=pad_nodes,
        pad_edges=pad_edges,
        graphs_seen=graphs_seen,
        kept_graphs=kept,
        max_single_nodes=max_single_nodes,
        max_single_edges=max_single_edges,
    )
    return _iter(), info


def _estimate_caps(
    datasets: list[HDF5Dataset],
    z_table,
    r_max: float,
) -> tuple[int, int]:
    converter = AtomsToGraphs(atomic_numbers=z_table, r_max=r_max)
    max_nodes = max_edges = 1
    for dataset in datasets:
        for idx in range(len(dataset)):
            atoms = dataset[idx]
            graph = converter.convert(atoms)
            nodes = int(graph.n_node.sum())
            edges = int(graph.n_edge.sum())
            max_nodes = max(max_nodes, nodes)
            max_edges = max(max_edges, edges)
    return max_nodes, max_edges
