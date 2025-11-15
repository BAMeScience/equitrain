from importlib import import_module

_BACKEND_MODULES = {
    'torch': 'equitrain.backends.torch_backend',
    'jax': 'equitrain.backends.jax_backend',
}


def get_backend(name: str):
    """
    Dynamically import and return a backend module.

    Parameters
    ----------
    name
        Backend identifier provided via CLI/args.

    Returns
    -------
    module
        The backend module exposing train/evaluate APIs.
    """
    try:
        module_path = _BACKEND_MODULES[name]
    except KeyError as exc:
        raise ValueError(f'Unknown backend "{name}"') from exc

    return import_module(module_path)
