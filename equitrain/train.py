import os
from pathlib import Path

from equitrain.backends import get_backend


def _resolve_input_path(path):
    if not path:
        return path

    if not isinstance(path, str | os.PathLike):
        return path

    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)

    repo_root = Path(__file__).resolve().parents[1]
    fallback = repo_root / 'tests' / candidate
    if fallback.exists():
        return str(fallback)

    return path


def train(args):
    """
    Dispatch training to the selected backend.

    Parameters
    ----------
    args
        Namespace returned by the CLI argument parser. Must provide a
        ``backend`` attribute (defaults to ``'torch'``).
    """
    backend_name = getattr(args, 'backend', 'torch') or 'torch'
    backend = get_backend(backend_name)

    if not hasattr(backend, 'train'):
        raise NotImplementedError(
            f'Backend "{backend_name}" does not expose a train() function.'
        )

    for attr in (
        'train_file',
        'valid_file',
        'test_file',
        'output_dir',
        'load_checkpoint',
        'load_checkpoint_model',
        'model',
    ):
        if hasattr(args, attr):
            setattr(args, attr, _resolve_input_path(getattr(args, attr)))

    return backend.train(args)


import sys as _sys

if 'equitrain' in _sys.modules:
    setattr(_sys.modules['equitrain'], 'train', train)
