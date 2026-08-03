from __future__ import annotations

import pytest
import torch

from equitrain.backends.torch_wrappers.mace import _ensure_data_get


def test_ensure_data_get_adds_mapping_get_to_mace_batch():
    pytest.importorskip('mace')

    from mace.tools import torch_geometric
    from mace.tools.torch_geometric.data import Data

    batch = torch_geometric.batch.Batch.from_data_list([Data(x=torch.ones(1, 1))])
    assert not hasattr(batch, 'get')

    returned = _ensure_data_get(batch)

    assert returned is batch
    assert batch.get('x').shape == (1, 1)
    assert batch.get('missing') is None
    sentinel = object()
    assert batch.get('missing', sentinel) is sentinel
