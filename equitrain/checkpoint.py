"""Backend-aware checkpoint helpers."""

from __future__ import annotations


def _resolve_backend(module_name: str, backend: str):
    if backend == 'jax':
        from importlib import import_module

        return import_module(f'equitrain.backends.jax_{module_name}')

    from importlib import import_module

    return import_module(f'equitrain.backends.torch_{module_name}')


def _infer_backend(args):
    if args:
        candidate = args[0]
        return getattr(candidate, 'backend', 'torch') or 'torch'
    return 'torch'


def load_model_state(*args, **kwargs):
    backend = kwargs.pop('backend', None) or 'torch'
    module = _resolve_backend('checkpoint', backend)
    return module.load_model_state(*args, **kwargs)


def load_checkpoint(*args, **kwargs):
    backend = kwargs.pop('backend', None) or _infer_backend(args)
    module = _resolve_backend('checkpoint', backend)
    return module.load_checkpoint(*args, **kwargs)


def save_checkpoint(*args, **kwargs):
    backend = kwargs.pop('backend', None) or _infer_backend(args)
    module = _resolve_backend('checkpoint', backend)
    return module.save_checkpoint(*args, **kwargs)


__all__ = ['load_model_state', 'load_checkpoint', 'save_checkpoint']
