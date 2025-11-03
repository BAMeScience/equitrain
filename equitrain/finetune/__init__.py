"""
Fine-tuning utilities shared across backends.

This package currently provides additive delta wrappers for both Torch and JAX
models, enabling lightweight fine-tuning workflows that keep the original
parameters frozen while optimising residual offsets.
"""

from .delta_torch import DeltaFineTuneWrapper as TorchDeltaFineTuneWrapper
from .delta_jax import (
    DeltaFineTuneModule as JaxDeltaFineTuneModule,
    ensure_delta_params as ensure_jax_delta_params,
    wrap_with_deltas as wrap_jax_module_with_deltas,
)

__all__ = [
    'TorchDeltaFineTuneWrapper',
    'JaxDeltaFineTuneModule',
    'ensure_jax_delta_params',
    'wrap_jax_module_with_deltas',
]
