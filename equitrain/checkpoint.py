"""Backend-agnostic checkpoint helpers."""

from __future__ import annotations


def load_model_state(*args, **kwargs):
    from equitrain.backends.torch_checkpoint import load_model_state as _impl

    return _impl(*args, **kwargs)


def load_checkpoint(*args, **kwargs):
    from equitrain.backends.torch_checkpoint import load_checkpoint as _impl

    return _impl(*args, **kwargs)


def save_checkpoint(*args, **kwargs):
    from equitrain.backends.torch_checkpoint import save_checkpoint as _impl

    return _impl(*args, **kwargs)


__all__ = ['load_model_state', 'load_checkpoint', 'save_checkpoint']
