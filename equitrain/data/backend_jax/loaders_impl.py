from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import jraph
import numpy as np

from equitrain.data.backend_jax.atoms_to_graphs import (
    AtomsToGraphs,
    graph_from_configuration,
)
from equitrain.data.configuration import Configuration, niggli_reduce_inplace
from equitrain.data.format_hdf5.dataset import HDF5Dataset


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

        def _iter_graphs():
            rng = np.random.default_rng(self._seed)
            if self._max_batches is None or self._n_graph is None:
                target = None
            else:
                target = self._max_batches * self._n_graph
            emitted = 0
            sources = list(self._datasets)
            if self._shuffle and len(sources) > 1:
                rng.shuffle(sources)
            for src in sources:
                ds = src
                indices = np.arange(len(ds))
                if self._shuffle and len(indices) > 1:
                    rng.shuffle(indices)
                for idx in indices:
                    atoms = ds[idx]
                    if self._niggli_reduce:
                        atoms = atoms.copy()
                        niggli_reduce_inplace(atoms)
                    conf = Configuration.from_atoms(atoms)
                    graph = graph_from_configuration(
                        conf, cutoff=float(r_max), z_table=z_table
                    )
                    yield graph
                    emitted += 1
                    if target is not None and emitted >= target:
                        return

        self._graph_iter_fn = _iter_graphs
        self._pending_info: dict | None = None

    def _pack(self):
        batches, info = pack_graphs_greedy(
            graph_iter_fn=self._graph_iter_fn,
            max_edges_per_batch=self._n_edge,
            max_nodes_per_batch=self._n_node,
            batch_size_limit=self._n_graph,
        )
        self._pack_info = info
        return batches, info

    def __iter__(self):
        batches, _ = self._pack()
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
