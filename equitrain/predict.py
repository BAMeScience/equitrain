"""Backend-agnostic prediction entry points."""

from __future__ import annotations


def _ensure_torch_backend(args):
    backend_name = getattr(args, 'backend', 'torch') or 'torch'
    if backend_name != 'torch':
        raise NotImplementedError(
            f'Prediction is not implemented for backend "{backend_name}".'
        )


def predict(args):
    _ensure_torch_backend(args)
    from equitrain.backends.torch_predict import predict as _predict

    return _predict(args)


def _predict(args, device=None):
    _ensure_torch_backend(args)
    from equitrain.backends.torch_predict import _predict as _impl

    return _impl(args, device=device)


def predict_graphs(*args, **kwargs):
    from equitrain.backends.torch_predict import predict_graphs as _impl

    return _impl(*args, **kwargs)


def predict_atoms(*args, **kwargs):
    from equitrain.backends.torch_predict import predict_atoms as _impl

    return _impl(*args, **kwargs)


def predict_structures(*args, **kwargs):
    from equitrain.backends.torch_predict import predict_structures as _impl

    return _impl(*args, **kwargs)


__all__ = [
    'predict',
    '_predict',
    'predict_graphs',
    'predict_atoms',
    'predict_structures',
]


import sys as _sys

if 'equitrain' in _sys.modules:
    _pkg = _sys.modules['equitrain']
    setattr(_pkg, 'predict', predict)
    setattr(_pkg, 'predict_atoms', predict_atoms)
    setattr(_pkg, 'predict_structures', predict_structures)
    setattr(_pkg, 'predict_graphs', predict_graphs)
