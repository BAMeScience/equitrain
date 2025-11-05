"""
Namespace package exposing JAX wrappers with lazy imports.
"""

from __future__ import annotations

from importlib import import_module
from types import MappingProxyType

__all__ = ['MaceWrapper', 'available_wrappers']

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
