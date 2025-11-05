from .argparser import (  # noqa: E402
    ArgumentError,
    check_args_complete,
    get_args_parser_evaluate,
    get_args_parser_export,
    get_args_parser_inspect,
    get_args_parser_predict,
    get_args_parser_preprocess,
    get_args_parser_train,
)


def train(args):
    from .train import train as _train

    return _train(args)


def evaluate(args):
    from .evaluate import evaluate as _evaluate

    return _evaluate(args)


def get_model(args, *extra_args, **extra_kwargs):
    """Backend-aware model loader.

    Parameters
    ----------
    args : Namespace
        Parsed CLI/pytest arguments that include at least a ``backend`` field.
    *extra_args, **extra_kwargs
        Forwarded to the backend specific loader when supported (Torch only for now).

    Returns
    -------
    object
        Torch: ``torch.nn.Module`` (possibly wrapped).
        JAX : ``ModelBundle`` as returned by ``equitrain.backends.jax_utils.load_model_bundle``.
    """

    backend = getattr(args, 'backend', 'torch') or 'torch'

    if backend == 'torch':
        from .backends.torch_model import get_model as _torch_get_model

        return _torch_get_model(args, *extra_args, **extra_kwargs)

    if backend == 'jax':
        from .backends.jax_utils import load_model_bundle

        model_path = getattr(args, 'model', None)
        if model_path is None:
            raise ValueError('JAX backend requires ``args.model`` to be set.')
        dtype = getattr(args, 'dtype', 'float32')
        return load_model_bundle(model_path, dtype=dtype)

    raise NotImplementedError(f"get_model is not implemented for backend '{backend}'.")


def preprocess(*args, **kwargs):
    from .preprocess import preprocess as _preprocess

    return _preprocess(*args, **kwargs)


def predict(*args, **kwargs):
    from .predict import predict as _predict

    return _predict(*args, **kwargs)


def predict_atoms(*args, **kwargs):
    from .predict import predict_atoms as _predict_atoms

    return _predict_atoms(*args, **kwargs)


def predict_structures(*args, **kwargs):
    from .predict import predict_structures as _predict_structures

    return _predict_structures(*args, **kwargs)


def predict_graphs(*args, **kwargs):
    from .predict import predict_graphs as _predict_graphs

    return _predict_graphs(*args, **kwargs)
