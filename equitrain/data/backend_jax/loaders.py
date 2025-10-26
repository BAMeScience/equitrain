from __future__ import annotations

from collections.abc import Iterable

import jraph
from mace_jax.data.utils import GraphDataLoader


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

    return max_nodes + 1, max_edges + 1


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

    pad_nodes, pad_edges = compute_padding_limits(graphs, max_nodes, max_edges)

    return GraphDataLoader(
        graphs=graphs,
        n_node=pad_nodes,
        n_edge=pad_edges,
        n_graph=max(batch_size, 2),
        shuffle=shuffle,
    )
