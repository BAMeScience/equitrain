from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def test_normalize_preserves_frozen_optimizer_parameter_trees():
    flax = pytest.importorskip('flax')
    jax = pytest.importorskip('jax')
    from equitrain.backends.jax_nnx_compat import normalize_pure_dict

    frozen = flax.core.freeze({'params': {'weight': [1.0]}})
    restored = normalize_pure_dict(frozen, {'params': {'weight': [2.0]}})
    assert isinstance(restored, flax.core.FrozenDict)
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(
        frozen
    )
    plain = normalize_pure_dict(
        {'params': {'weight': [1.0]}}, {'params': {'weight': [2.0]}}
    )
    assert type(plain) is dict


def test_jax_nnx_compat_imports_without_mace_jax(monkeypatch):
    real_import = builtins.__import__
    previous_module = sys.modules.pop('equitrain.backends.jax_nnx_compat', None)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'mace_jax' or name.startswith('mace_jax.'):
            raise ModuleNotFoundError("No module named 'mace_jax'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    try:
        compat = importlib.import_module('equitrain.backends.jax_nnx_compat')
        restored = compat.normalize_pure_dict(
            {1: (0, {'nested': [1]})},
            {'1': [2, {'nested': [3]}]},
        )
        assert restored == {1: (2, {'nested': [3]})}
    finally:
        sys.modules.pop('equitrain.backends.jax_nnx_compat', None)
        if previous_module is not None:
            sys.modules['equitrain.backends.jax_nnx_compat'] = previous_module
