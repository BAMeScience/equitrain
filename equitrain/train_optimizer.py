"""Compatibility shims for torch optimizer helpers."""

from __future__ import annotations


def update_weight_decay(*args, **kwargs):
    from equitrain.backends.torch_optimizer import update_weight_decay as _impl

    return _impl(*args, **kwargs)


def add_weight_decay(*args, **kwargs):
    from equitrain.backends.torch_optimizer import add_weight_decay as _impl

    return _impl(*args, **kwargs)


def optimizer_kwargs(*args, **kwargs):
    from equitrain.backends.torch_optimizer import optimizer_kwargs as _impl

    return _impl(*args, **kwargs)


def create_optimizer(*args, **kwargs):
    from equitrain.backends.torch_optimizer import create_optimizer as _impl

    return _impl(*args, **kwargs)


def create_optimizer_impl(*args, **kwargs):
    from equitrain.backends.torch_optimizer import create_optimizer_impl as _impl

    return _impl(*args, **kwargs)


__all__ = [
    'update_weight_decay',
    'add_weight_decay',
    'optimizer_kwargs',
    'create_optimizer',
    'create_optimizer_impl',
]
