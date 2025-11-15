"""Test utilities that wrap optional model backends.

Wrappers depend on heavy optional requirements. Import each lazily so missing
packages do not break unrelated tests (e.g., running M3GNet without MACE).
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    'AniWrapper',
    'MaceWrapper',
    'M3GNetWrapper',
    'OrbWrapper',
    'SevennetWrapper',
]


def _optional_import(module: str, attr: str):
    try:
        return getattr(import_module(module, __name__), attr)
    except Exception:  # pragma: no cover - optional dependency guard
        return None


AniWrapper = _optional_import('.wrapper_ani', 'AniWrapper')
MaceWrapper = _optional_import('.wrapper_mace', 'MaceWrapper')
M3GNetWrapper = _optional_import('.wrapper_m3gnet', 'M3GNetWrapper')
OrbWrapper = _optional_import('.wrapper_orb', 'OrbWrapper')
SevennetWrapper = _optional_import('.wrapper_sevennet', 'SevennetWrapper')


def __getattr__(name: str):
    if name in __all__:
        value = globals().get(name)
        if value is None:
            raise ImportError(
                f'Wrapper {name!r} is unavailable because its optional dependency '
                'is not installed.'
            )
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
