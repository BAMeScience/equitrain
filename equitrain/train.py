from equitrain.backends import get_backend


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

    return backend.train(args)


import sys as _sys

if 'equitrain' in _sys.modules:
    setattr(_sys.modules['equitrain'], 'train', train)
