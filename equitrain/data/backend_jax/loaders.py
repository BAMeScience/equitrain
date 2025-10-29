from __future__ import annotations

import math
from collections.abc import Iterable

import jax
import jax.numpy as jnp
import jraph
import numpy as np


def _pad_array(array: jnp.ndarray, pad: int) -> jnp.ndarray:
    if pad <= 0:
        return array
    pad_config = [(0, pad)] + [(0, 0)] * (array.ndim - 1)
    return jnp.pad(array, pad_config, mode='constant', constant_values=0)


def _pad_pytree(tree, pad: int):
    if tree is None or pad <= 0:
        return tree
    return jax.tree_util.tree_map(lambda x: _pad_array(jnp.asarray(x), pad), tree)


def _pad_single_graph(
    graph: jraph.GraphsTuple, target_nodes: int, target_edges: int
) -> jraph.GraphsTuple:
    actual_nodes = int(graph.n_node.sum())
    actual_edges = int(graph.n_edge.sum())

    if actual_nodes > target_nodes or actual_edges > target_edges:
        return graph

    pad_nodes = target_nodes - actual_nodes
    pad_edges = target_edges - actual_edges

    nodes = _pad_pytree(graph.nodes, pad_nodes)
    edges = _pad_pytree(graph.edges, pad_edges)

    senders = jnp.pad(graph.senders, (0, pad_edges), constant_values=0)
    receivers = jnp.pad(graph.receivers, (0, pad_edges), constant_values=0)

    return graph._replace(nodes=nodes, edges=edges, senders=senders, receivers=receivers)


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
    ) -> None:
        self._graphs = list(graphs)
        self._n_node = int(max(n_node, 0))
        self._n_edge = int(max(n_edge, 0))
        self._n_graph = max(int(n_graph), 1)
        self._shuffle = shuffle

    def __iter__(self):
        if not self._graphs:
            return

        indices = np.arange(len(self._graphs))
        if self._shuffle and len(indices) > 1:
            np.random.default_rng().shuffle(indices)

        for start in range(0, len(indices), self._n_graph):
            batch_indices = indices[start : start + self._n_graph]
            batch_graphs = [self._graphs[i] for i in batch_indices]

            if self._n_graph == 1:
                graph = batch_graphs[0]
                yield _pad_single_graph(graph, self._n_node, self._n_edge)
                continue

            batched = jraph.batch_np(batch_graphs)
            batched = jraph.pad_with_graphs(
                batched,
                n_node=self._n_node,
                n_edge=self._n_edge,
                n_graph=self._n_graph,
            )
            yield batched

    def __len__(self) -> int:
        if not self._graphs:
            return 0
        return math.ceil(len(self._graphs) / self._n_graph)


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

    return GraphDataLoader(
        graphs=graphs,
        n_node=pad_nodes,
        n_edge=pad_edges,
        n_graph=max(int(batch_size), 1),
        shuffle=shuffle,
    )
