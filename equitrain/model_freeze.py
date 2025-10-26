"""Compatibility wrapper for legacy imports.

The PyTorch-specific implementation now lives in ``equitrain.backends.torch_freeze``.
"""

from equitrain.backends.torch_freeze import model_freeze_params  # noqa: F401
