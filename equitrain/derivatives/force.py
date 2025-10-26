"""Compatibility shim for torch force derivatives."""

from __future__ import annotations


def compute_force(*args, **kwargs):
    from equitrain.backends.torch_derivatives.force import compute_force as _impl

    return _impl(*args, **kwargs)


__all__ = ['compute_force']
