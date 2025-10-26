"""Backend-agnostic model loader shim."""

from __future__ import annotations


def get_model(args, logger=None):
    backend_name = getattr(args, 'backend', 'torch') or 'torch'
    if backend_name != 'torch':
        raise NotImplementedError(
            f'No model loader implemented for backend "{backend_name}".'
        )

    from equitrain.backends.torch_model import get_model as _get_model

    return _get_model(args, logger=logger)


__all__ = ['get_model']
