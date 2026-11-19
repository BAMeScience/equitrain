"""Native GRACE-1L (Academic Software Licence; see LICENSE.md).

Backends are imported lazily so JAX-only installations do not require Torch.
"""

from .config import GraceConfig

__all__ = ['Grace1L', 'GraceConfig', 'JaxGrace1L']


def __getattr__(name):
    if name == 'Grace1L':
        from .model import Grace1L

        return Grace1L
    if name == 'JaxGrace1L':
        from .jax_model import JaxGrace1L

        return JaxGrace1L
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
