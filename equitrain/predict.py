"""Backend-agnostic prediction entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _select_backend(args):
    backend_name = getattr(args, 'backend', 'torch') or 'torch'
    if backend_name not in {'torch', 'jax'}:
        raise NotImplementedError(
            f'Prediction is not implemented for backend "{backend_name}".'
        )
    return backend_name


def _array_or_none(value: Any):
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _write_prediction_results(args, backend_name: str, predictions) -> None:
    output_dir = getattr(args, 'output_dir', None)
    if not output_dir:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    names = ('energy', 'forces', 'stress')
    arrays = {}
    metadata_arrays = {}
    for name, value in zip(names, predictions, strict=True):
        array = _array_or_none(value)
        if array is None:
            metadata_arrays[name] = None
            continue
        arrays[name] = array
        metadata_arrays[name] = {
            'shape': list(array.shape),
            'dtype': str(array.dtype),
        }

    arrays_path = output_path / 'predictions.npz'
    np.savez(arrays_path, **arrays)

    metadata_path = output_path / 'predictions.json'
    payload = {
        'backend': backend_name,
        'dataset': getattr(args, 'predict_file', None),
        'arrays_file': arrays_path.name,
        'arrays': metadata_arrays,
    }
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def predict(args):
    backend_name = _select_backend(args)
    if backend_name == 'torch':
        from equitrain.backends.torch_predict import predict as _predict_impl
    else:
        from equitrain.backends.jax_predict import predict as _predict_impl

    predictions = _predict_impl(args)
    _write_prediction_results(args, backend_name, predictions)
    return predictions


def _predict(args, device=None):
    backend_name = _select_backend(args)
    if backend_name != 'torch':
        raise NotImplementedError(
            '_predict with explicit device control is only available for the Torch backend.'
        )
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
