"""
Namespace package that exposes torch wrappers with lazy optional imports.
"""

from __future__ import annotations

from importlib import import_module
from types import MappingProxyType

from .base import AbstractWrapper

__all__ = [
    'AbstractWrapper',
    'MaceWrapper',
    'SevennetWrapper',
    'OrbWrapper',
    'AniWrapper',
    'M3GNetWrapper',
    'available_wrappers',
]

_MODULE_MAP: dict[str, str] = {
    'MaceWrapper': 'mace',
    'SevennetWrapper': 'sevennet',
    'OrbWrapper': 'orb',
    'AniWrapper': 'ani',
    'M3GNetWrapper': 'm3gnet',
}

_CACHE: dict[str, object] = {}
_ERRORS: dict[str, Exception] = {}


def __getattr__(name: str):
    if name == 'AbstractWrapper':
        return AbstractWrapper
    if name not in _MODULE_MAP:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

    if name in _CACHE:
        return _CACHE[name]

    try:
        module = import_module(f'.{_MODULE_MAP[name]}', __name__)
        attr = getattr(module, name)
    except Exception as exc:  # pragma: no cover - propagate original error
        _ERRORS[name] = exc
        raise

    _CACHE[name] = attr
    return attr


def available_wrappers() -> MappingProxyType:
    """
    Return a mapping of wrapper name -> availability (True or Exception).
    """
    status: dict[str, object] = {}
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
        except Exception as exc:  # pragma: no cover - optional import errors
            _ERRORS[name] = exc
            status[name] = exc
        else:
            status[name] = True
    return MappingProxyType(status)
