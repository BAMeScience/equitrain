"""
Fine-tuning utilities shared across backends.

This package currently provides additive delta and LoRA wrappers for both
Torch and JAX models, enabling lightweight fine-tuning workflows that keep the
original parameters frozen while optimising small residual adapters.
"""

from .delta_jax import (
    DeltaFineTuneModule as JaxDeltaFineTuneModule,
)
from .delta_jax import (
    ensure_delta_params as ensure_jax_delta_params,
)
from .delta_jax import (
    wrap_with_deltas as wrap_jax_module_with_deltas,
)
from .delta_torch import DeltaFineTuneWrapper as TorchDeltaFineTuneWrapper
from .lora_jax import (
    LoRAFineTuneModule as JaxLoRAFineTuneModule,
)
from .lora_jax import (
    ensure_lora_params as ensure_jax_lora_params,
)
from .lora_jax import (
    wrap_with_lora as wrap_jax_module_with_lora,
)
from .lora_torch import LoRAFineTuneWrapper as TorchLoRAFineTuneWrapper

__all__ = [
    'TorchDeltaFineTuneWrapper',
    'JaxDeltaFineTuneModule',
    'ensure_jax_delta_params',
    'wrap_jax_module_with_deltas',
    'TorchLoRAFineTuneWrapper',
    'JaxLoRAFineTuneModule',
    'ensure_jax_lora_params',
    'wrap_jax_module_with_lora',
]
