from types import SimpleNamespace

import jax.numpy as jnp
import jraph
import numpy as np
import torch.serialization

torch.serialization.add_safe_globals([slice])

from equitrain import get_args_parser_predict
from equitrain.backends import jax_predict


def _dummy_graph(*, padded: bool = False):
    n_node = jnp.array([1], dtype=jnp.int32)
    n_edge = jnp.array([0], dtype=jnp.int32)
    if padded:
        n_node = jnp.array([0], dtype=jnp.int32)
    return jraph.GraphsTuple(
        nodes={'positions': jnp.zeros((int(n_node.sum()) or 1, 3))},
        edges={'shifts': jnp.zeros((int(n_edge.sum()), 3))},
        senders=jnp.zeros((int(n_edge.sum()),), dtype=jnp.int32),
        receivers=jnp.zeros((int(n_edge.sum()),), dtype=jnp.int32),
        globals={'cell': jnp.zeros((3, 3))},
        n_node=n_node,
        n_edge=n_edge,
    )


def test_jax_predict_basic(monkeypatch):
    records = {'graphs': []}

    def fake_load_model_bundle(path, dtype=None, wrapper=None):
        records['bundle_path'] = path
        records['bundle_wrapper'] = wrapper
        return SimpleNamespace(
            params={'weights': 1.0},
            module=None,
            config={'atomic_numbers': [1], 'r_max': 2.5},
        )

    monkeypatch.setattr(jax_predict, '_load_bundle', fake_load_model_bundle)

    def fake_get_dataloader(*args, **kwargs):
        records['loader_kwargs'] = kwargs
        # second graph simulates padding (n_node=0)
        return [_dummy_graph(), _dummy_graph(padded=True)]
