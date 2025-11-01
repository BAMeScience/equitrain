import numpy as np
import jax.numpy as jnp
import jraph

from equitrain.data.backend_jax.loaders import GraphDataLoader, build_loader


def _make_graph(graph_id: int) -> jraph.GraphsTuple:
    node_feat = jnp.array([[float(graph_id)]], dtype=jnp.float32)
    return jraph.GraphsTuple(
        nodes=node_feat,
        edges=jnp.zeros((0, 1), dtype=jnp.float32),
        senders=jnp.array([], dtype=np.int32),
        receivers=jnp.array([], dtype=np.int32),
        n_node=jnp.array([1], dtype=np.int32),
        n_edge=jnp.array([0], dtype=np.int32),
        globals=jnp.array([float(graph_id)], dtype=jnp.float32),
    )


def _collect_sequence(loader: GraphDataLoader) -> list[float]:
    sequence: list[float] = []
    for graph in loader:
        globals_array = np.asarray(graph.globals)
        sequence.append(float(globals_array.reshape(-1)[0]))
    return sequence


def test_graph_data_loader_deterministic_shuffle():
    graphs = [_make_graph(i) for i in range(6)]

    loader_a = build_loader(
        graphs,
        batch_size=1,
        shuffle=True,
        max_nodes=None,
        max_edges=None,
        seed=2025,
    )
    loader_b = build_loader(
        graphs,
        batch_size=1,
        shuffle=True,
        max_nodes=None,
        max_edges=None,
        seed=2025,
    )
    loader_c = build_loader(
        graphs,
        batch_size=1,
        shuffle=True,
        max_nodes=None,
        max_edges=None,
        seed=1337,
    )

    seq_a = _collect_sequence(loader_a)
    seq_b = _collect_sequence(loader_b)
    seq_c = _collect_sequence(loader_c)

    assert seq_a == seq_b
    assert seq_a != seq_c
