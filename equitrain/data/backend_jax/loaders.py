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

    def __iter__(self):
        if not self._graphs:
            return

        indices = np.arange(len(self._graphs))
        if self._shuffle and len(indices) > 1:
            self._rng.shuffle(indices)

        for start in range(0, len(indices), self._n_graph):
            batch_indices = indices[start : start + self._n_graph]
            batch_graphs = [self._graphs[i] for i in batch_indices]

            if self._n_graph == 1 and self._max_nodes_per_batch is None and self._max_edges_per_batch is None:
                yield batch_graphs[0]
                continue

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
                (self._max_nodes_per_batch is not None and nodes > self._max_nodes_per_batch)
                or (self._max_edges_per_batch is not None and edges > self._max_edges_per_batch)
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
                micro_batches.append(self._finalize_batch(current_graphs))
                current_graphs = []
                current_nodes = 0
                current_edges = 0

            if (
                self._max_nodes_per_batch is not None and nodes > self._max_nodes_per_batch
            ) or (
                self._max_edges_per_batch is not None and edges > self._max_edges_per_batch
            ):
                micro_batches.append(self._finalize_batch([graph]))
                continue

            current_graphs.append(graph)
            current_nodes += nodes
            current_edges += edges

        if current_graphs:
            micro_batches.append(self._finalize_batch(current_graphs))

        return micro_batches

    def _finalize_batch(self, graphs: list[jraph.GraphsTuple]) -> jraph.GraphsTuple:

        if not graphs:
            raise ValueError('Cannot finalize an empty batch of graphs.')

        if len(graphs) == 1:
            batched = graphs[0]
        else:
            batched = jraph.batch_np(graphs)

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

    if max_nodes_override is not None:
        max_nodes = min(max_nodes, max_nodes_override)
    if max_edges_override is not None:
        max_edges = min(max_edges, max_edges_override)

    return max_nodes, max_edges


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
) -> GraphDataLoader | None:
    if not graphs:
        return None

    max_nodes_actual, max_edges_actual = compute_padding_limits(
        graphs, max_nodes, max_edges
    )

    pad_nodes = max_nodes_actual
    pad_edges = max_edges_actual

    if max_nodes is not None:
        pad_nodes = min(pad_nodes, max_nodes)
    if max_edges is not None:
        pad_edges = min(pad_edges, max_edges)

    if int(batch_size) > 1:
        pad_nodes += 1
        pad_edges += 1

    if rng is None and seed is not None:
        rng = np.random.default_rng(seed)

    return GraphDataLoader(
        graphs=graphs,
        n_node=pad_nodes,
        n_edge=pad_edges,
        n_graph=max(int(batch_size), 1),
        shuffle=shuffle,
        max_nodes_per_batch=max_nodes,
        max_edges_per_batch=max_edges,
        drop_oversized=drop,
        rng=rng,
    )
