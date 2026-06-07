"""
Fine-tuning utilities shared across backends.

This package provides additive delta and LoRA wrappers for Torch and JAX models.
Exports are imported lazily so installing one backend does not require the other.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    'TorchDeltaFineTuneWrapper': ('.delta_torch', 'DeltaFineTuneWrapper'),
    'JaxDeltaFineTuneModule': ('.delta_jax', 'DeltaFineTuneModule'),
    'ensure_jax_delta_params': ('.delta_jax', 'ensure_delta_params'),
    'wrap_jax_module_with_deltas': ('.delta_jax', 'wrap_with_deltas'),
    'TorchLoRAFineTuneWrapper': ('.lora_torch', 'LoRAFineTuneWrapper'),
    'JaxLoRAFineTuneModule': ('.lora_jax', 'LoRAFineTuneModule'),
    'ensure_jax_lora_params': ('.lora_jax', 'ensure_lora_params'),
    'wrap_jax_module_with_lora': ('.lora_jax', 'wrap_with_lora'),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from exc

    module = import_module(module_name, __name__)
    attr = getattr(module, attr_name)
    globals()[name] = attr
    return attr
