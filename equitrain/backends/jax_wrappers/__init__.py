"""
Namespace package exposing JAX wrappers with lazy imports.
"""

from __future__ import annotations

from importlib import import_module
from types import MappingProxyType

__all__ = ['MaceWrapper', 'available_wrappers', 'get_wrapper_builder']

_MODULE_MAP = {'MaceWrapper': 'mace'}
_CACHE = {}
_ERRORS = {}


def __getattr__(name: str):
    if name not in _MODULE_MAP:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

    if name in _CACHE:
        return _CACHE[name]

    try:
        module = import_module(f'.{_MODULE_MAP[name]}', __name__)
        attr = getattr(module, name)
    except Exception as exc:  # pragma: no cover
        _ERRORS[name] = exc
        raise

    _CACHE[name] = attr
    return attr


def available_wrappers() -> MappingProxyType:
    status = {}
    for name, module_name in _MODULE_MAP.items():
        if name in _CACHE:
            status[name] = True
            continue
        if name in _ERRORS:
            status[name] = _ERRORS[name]
            continue
        try:
            module = import_module(f'.{module_name}', __name__)
            getattr(module, name)
        except Exception as exc:  # pragma: no cover
            _ERRORS[name] = exc
            status[name] = exc
        else:
            status[name] = True
    return MappingProxyType(status)


def get_wrapper_builder(name: str):
    """
    Return the ``build_module`` helper for the requested wrapper.
    """

    if not name:
        raise ValueError('Wrapper name must be provided to load a JAX model.')

    module_name = name.strip().lower()
    try:
        module = import_module(f'.{module_name}', __name__)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f'Unknown JAX wrapper "{name}". Expected module "{__name__}.{module_name}".'
        ) from exc

    builder = getattr(module, 'build_module', None)
    if builder is None:
        raise AttributeError(
            f'Wrapper "{name}" does not expose a build_module(config) helper.'
        )
    return builder
