try:  # pragma: no cover - lightweight import guard
    import torch.serialization as _torch_serialization  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    _torch_serialization = None
else:  # pragma: no branch
    add_safe_globals = getattr(_torch_serialization, 'add_safe_globals', None)
    if callable(add_safe_globals):
        try:
            add_safe_globals([slice])
        except Exception:  # noqa: BLE001
            pass

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


def get_model(*args, **kwargs):
    from .model import get_model as _get_model

    return _get_model(*args, **kwargs)


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
