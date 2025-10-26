"""Compatibility shim for torch stress derivatives."""

from __future__ import annotations


def compute_stress(*args, **kwargs):
    from equitrain.backends.torch_derivatives.stress import compute_stress as _impl

    return _impl(*args, **kwargs)


def get_displacement(*args, **kwargs):
    from equitrain.backends.torch_derivatives.stress import get_displacement as _impl

    return _impl(*args, **kwargs)


__all__ = ['compute_stress', 'get_displacement']
