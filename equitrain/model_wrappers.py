"""
Compatibility re-export of torch wrappers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from equitrain.backends import torch_wrappers as _torch_wrappers

__all__ = list(getattr(_torch_wrappers, '__all__', ()))

if TYPE_CHECKING:
    from equitrain.backends.torch_wrappers import (
        AbstractWrapper,
        AniWrapper,
        M3GNetWrapper,
        MaceWrapper,
        OrbWrapper,
        SevennetWrapper,
    )
    from equitrain.backends.torch_wrappers import (
        available_wrappers as _available_wrappers_type,
    )


def __getattr__(name: str):
    if name in __all__:
        return getattr(_torch_wrappers, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def available_wrappers():
    return _torch_wrappers.available_wrappers()
