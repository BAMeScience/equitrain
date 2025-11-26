from __future__ import annotations

import math
from collections.abc import Iterable

import jraph
import numpy as np


class GraphDataLoader:
    """Lightweight graph data loader compatible with JAX/Flax training loops."""

    def __init__(
        self,
        *,
        graphs: list[jraph.GraphsTuple],
        n_node: int,
        n_edge: int,
        n_graph: int,
        shuffle: bool = True,
        max_nodes_per_batch: int | None = None,
        max_edges_per_batch: int | None = None,
        drop_oversized: bool = False,
        rng: np.random.Generator | None = None,
        force_padding: bool = False,
        pad_total_nodes: int | None = None,
        pad_total_edges: int | None = None,
    ) -> None:
        self._graphs = list(graphs)
        self._n_node = int(max(n_node, 0))
        self._n_edge = int(max(n_edge, 0))
        self._n_graph = max(int(n_graph), 1)
        self._shuffle = shuffle
        self._rng = rng if rng is not None else np.random.default_rng()
        self._max_nodes_per_batch = (
            int(max_nodes_per_batch) if max_nodes_per_batch is not None else None
        )
        self._max_edges_per_batch = (
            int(max_edges_per_batch) if max_edges_per_batch is not None else None
        )
        self._drop_oversized = bool(drop_oversized)
        self._force_padding = bool(force_padding)
        self._pad_total_nodes = (
            int(pad_total_nodes) if pad_total_nodes is not None else None
        )
        self._pad_total_edges = (
            int(pad_total_edges) if pad_total_edges is not None else None
        )

    def __iter__(self):
        if not self._graphs:
            return

        indices = np.arange(len(self._graphs))
        if self._shuffle and len(indices) > 1:
            self._rng.shuffle(indices)

        for start in range(0, len(indices), self._n_graph):
            batch_indices = indices[start : start + self._n_graph]
            batch_graphs = [self._graphs[i] for i in batch_indices]

            micro_batches = self._build_micro_batches(batch_graphs)
            if not micro_batches:
                continue

            if len(micro_batches) == 1:
                yield micro_batches[0]
            else:
                yield micro_batches

    def __len__(self) -> int:
        if not self._graphs:
            return 0
        return math.ceil(len(self._graphs) / self._n_graph)

    def _build_micro_batches(
        self, batch_graphs: list[jraph.GraphsTuple]
    ) -> list[jraph.GraphsTuple]:
        micro_batches: list[jraph.GraphsTuple] = []
        current_graphs: list[jraph.GraphsTuple] = []
        current_nodes = 0
        current_edges = 0

        for graph in batch_graphs:
            nodes = int(graph.n_node.sum())
            edges = int(graph.n_edge.sum())

            if self._drop_oversized and (
                (
                    self._max_nodes_per_batch is not None
                    and nodes > self._max_nodes_per_batch
                )
                or (
                    self._max_edges_per_batch is not None
                    and edges > self._max_edges_per_batch
                )
            ):
                continue

            exceeds_nodes = (
                self._max_nodes_per_batch is not None
                and current_graphs
                and current_nodes + nodes > self._max_nodes_per_batch
            )
            exceeds_edges = (
                self._max_edges_per_batch is not None
                and current_graphs
                and current_edges + edges > self._max_edges_per_batch
            )

            if exceeds_nodes or exceeds_edges:
                finalized = self._finalize_batch(current_graphs)
                if finalized is not None:
                    micro_batches.append(finalized)
                current_graphs = []
                current_nodes = 0
                current_edges = 0

            if (
                self._max_nodes_per_batch is not None
                and nodes > self._max_nodes_per_batch
            ) or (
                self._max_edges_per_batch is not None
                and edges > self._max_edges_per_batch
            ):
                finalized = self._finalize_batch([graph])
                if finalized is not None:
                    micro_batches.append(finalized)
                continue

            current_graphs.append(graph)
            current_nodes += nodes
            current_edges += edges

        if current_graphs:
            finalized = self._finalize_batch(current_graphs)
            if finalized is not None:
                micro_batches.append(finalized)

        return micro_batches

    def _finalize_batch(self, graphs: list[jraph.GraphsTuple]) -> jraph.GraphsTuple:
        if not graphs:
            raise ValueError('Cannot finalize an empty batch of graphs.')

        if len(graphs) == 1:
            batched = graphs[0]
        else:
            batched = jraph.batch_np(graphs)

        if self._force_padding:
            # Pad every batch to fixed totals to ensure a single XLA shape.
            pad_graphs = self._n_graph  # fixed graph count
            target_nodes = (
                self._pad_total_nodes
                if self._pad_total_nodes is not None
                else self._n_node * pad_graphs
            )
            target_edges = (
                self._pad_total_edges
                if self._pad_total_edges is not None
                else self._n_edge * pad_graphs
            )
            return jraph.pad_with_graphs(
                batched,
                n_node=target_nodes,
                n_edge=target_edges,
                n_graph=pad_graphs,
            )

        needs_padding = (
            batched.n_node.shape[0] < self._n_graph
            or int(np.max(batched.n_node)) > self._n_node
            or int(np.max(batched.n_edge)) > self._n_edge
        )
        if not needs_padding:
            return batched

        padding_graphs = max(self._n_graph, batched.n_node.shape[0] + 1)
        target_nodes = max(self._n_node, int(np.max(batched.n_node))) * padding_graphs
        target_edges = max(self._n_edge, int(np.max(batched.n_edge))) * padding_graphs

        return jraph.pad_with_graphs(
            batched,
            n_node=target_nodes,
            n_edge=target_edges,
            n_graph=padding_graphs,
        )


def compute_padding_limits(
    graphs: Iterable[jraph.GraphsTuple],
    max_nodes_override: int | None,
    max_edges_override: int | None,
) -> tuple[int, int]:
    max_nodes = 0
    max_edges = 0
    for graph in graphs:
        max_nodes = max(max_nodes, int(graph.n_node.sum()))
        max_edges = max(max_edges, int(graph.n_edge.sum()))

    return max_nodes, max_edges


def pack_graphs_greedy(
    graphs: list[jraph.GraphsTuple],
    *,
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

    def _scan():
        current_len = edge_sum = node_sum = 0
        max_graphs = max_total_nodes = max_total_edges = 0
        dropped = batches = 0
        for g in graphs:
            g_edges = int(g.n_edge.sum())
            g_nodes = int(g.n_node.sum())
            if g_edges > max_edges_per_batch or (
                max_nodes_per_batch is not None and g_nodes > max_nodes_per_batch
            ):
                dropped += 1
                continue
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
        )

    pad_graphs, pad_nodes, pad_edges, dropped, total_batches = _scan()

    def _iter():
        current: list[jraph.GraphsTuple] = []
        edge_sum = node_sum = 0
        for g in graphs:
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
    )
    return _iter(), info


def build_loader(
    graphs: list[jraph.GraphsTuple],
    *,
    batch_size: int,
    shuffle: bool,
    max_nodes: int | None,
    max_edges: int | None,
    drop: bool = False,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    force_padding: bool = False,
    pad_total_nodes: int | None = None,
    pad_total_edges: int | None = None,
) -> GraphDataLoader | None:
    if not graphs:
        return None

    max_nodes_actual, max_edges_actual = compute_padding_limits(
        graphs, max_nodes, max_edges
    )

    # Use caller-provided caps as target padding, but never below actual maxima.
    pad_nodes = max_nodes if max_nodes is not None else max_nodes_actual
    pad_edges = max_edges if max_edges is not None else max_edges_actual
    # Ensure padding targets are never below the actual dataset maxima to avoid under-padding.
    pad_nodes = max(pad_nodes, max_nodes_actual)
    pad_edges = max(pad_edges, max_edges_actual)

    n_graph = max(int(batch_size), 1)
    pad_total_nodes = (
        int(pad_total_nodes) if pad_total_nodes is not None else pad_nodes * n_graph
    )
    pad_total_edges = (
        int(pad_total_edges) if pad_total_edges is not None else pad_edges * n_graph
    )

    if rng is None and seed is not None:
        rng = np.random.default_rng(seed)

    return GraphDataLoader(
        graphs=graphs,
        n_node=pad_nodes,
        n_edge=pad_edges,
        n_graph=n_graph,
        shuffle=shuffle,
        max_nodes_per_batch=None if force_padding else pad_total_nodes,
        max_edges_per_batch=None if force_padding else pad_total_edges,
        drop_oversized=drop,
        rng=rng,
        force_padding=force_padding,
        pad_total_nodes=pad_total_nodes if force_padding else None,
        pad_total_edges=pad_total_edges if force_padding else None,
    )
