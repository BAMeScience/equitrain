"""PyTorch-specific helpers for freezing/unfreezing model parameters."""

from __future__ import annotations

import re
from collections.abc import Iterable


def model_freeze_params(args, model, logger=None) -> None:
    """
    Freeze/unfreeze model parameters based on CLI regex patterns.

    This mirrors the historical behavior that lived in ``equitrain.model_freeze``.
    """

    def _matches(name: str, patterns: Iterable[str]) -> bool:
        return any(re.fullmatch(pattern, name) for pattern in patterns)

    patterns_unfreeze = getattr(args, 'unfreeze_params', None)
    patterns_freeze = getattr(args, 'freeze_params', None)

    if patterns_unfreeze:
        for name, param in model.named_parameters():
            param.requires_grad = _matches(name, patterns_unfreeze)
            if logger is not None:
                message = 'Unfreezing' if param.requires_grad else 'Freezing'
                logger.log(1, f'{message} parameter: {name}')
        return

    if patterns_freeze:
        for name, param in model.named_parameters():
            if _matches(name, patterns_freeze):
                param.requires_grad = False
                if logger is not None:
                    logger.log(1, f'Freezing parameter: {name}')
