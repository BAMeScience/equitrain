from types import SimpleNamespace

import jax.numpy as jnp
from flax import traverse_util
from flax.core import freeze

from equitrain.backends.jax_freeze import build_trainable_mask


def _flatten_bool_mask(mask):
    flat = traverse_util.flatten_dict(mask, sep='.')
    result = {}
    for key, value in flat.items():
        if isinstance(key, tuple):
            name = '.'.join(key)
        else:
            name = key
        result[name] = bool(value)
    return result


def _make_dummy_params():
    return freeze(
        {
            'params': {
                'layer1': {
                    'weight': jnp.ones((2, 2)),
                    'bias': jnp.zeros((2,)),
                },
                'layer2': {
                    'weight': jnp.full((3, 3), 2.0),
                    'bias': jnp.full((3,), -1.0),
                },
            }
        }
    )


def test_build_trainable_mask_default_false_freezes_all():
    params = _make_dummy_params()
    args = SimpleNamespace(unfreeze_params=None, freeze_params=None)

    mask = build_trainable_mask(
        args,
        params,
        default_trainable=False,
    )

    assert mask is not None
    flat_mask = _flatten_bool_mask(mask)
    assert all(trainable is False for trainable in flat_mask.values())


def test_build_trainable_mask_unfreeze_pattern_enables_subset():
    params = _make_dummy_params()
    args = SimpleNamespace(
        unfreeze_params=[r'layer1\.weight'],
        freeze_params=None,
    )

    mask = build_trainable_mask(
        args,
        params,
        default_trainable=False,
    )

    assert mask is not None
    flat_mask = _flatten_bool_mask(mask)
    assert flat_mask['params.layer1.weight'] is True
    assert flat_mask['params.layer1.bias'] is False
    assert flat_mask['params.layer2.weight'] is False
    assert flat_mask['params.layer2.bias'] is False


def test_build_trainable_mask_freeze_pattern_disables_subset():
    params = _make_dummy_params()
    args = SimpleNamespace(
        unfreeze_params=None,
        freeze_params=[r'layer2\.bias'],
    )

    mask = build_trainable_mask(
        args,
        params,
        default_trainable=True,
    )

    assert mask is not None
    flat_mask = _flatten_bool_mask(mask)
    assert flat_mask['params.layer2.bias'] is False
    assert flat_mask['params.layer1.weight'] is True
    assert flat_mask['params.layer1.bias'] is True
    assert flat_mask['params.layer2.weight'] is True
