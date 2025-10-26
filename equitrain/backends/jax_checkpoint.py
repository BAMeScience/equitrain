"""Minimal checkpoint helpers for the JAX backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from equitrain.logger import FileLogger


def load_model_state(*args, **kwargs):
    raise NotImplementedError(
        'Model state loading is not implemented for the JAX backend yet.'
    )


def load_checkpoint(
    args,
    model: Any,
    model_ema=None,
    accelerator=None,
    logger: FileLogger | None = None,
):
    # JAX training does not currently support checkpoint restore.
    if logger is not None:
        logger.log(1, 'Checkpoint loading is not implemented for the JAX backend.')
    return False, None


def save_checkpoint(*args, **kwargs):
    raise NotImplementedError(
        'Checkpoint saving is not implemented for the JAX backend. '
        'The backend writes parameters via ``jax_backend._save_parameters`` instead.'
    )


__all__ = ['load_model_state', 'load_checkpoint', 'save_checkpoint']
