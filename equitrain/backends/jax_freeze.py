"""JAX utilities for freezing/unfreezing parameters via regex patterns."""

from __future__ import annotations

import re
from typing import Iterable

from flax import traverse_util
from flax.core import frozen_dict
from flax.core.frozen_dict import FrozenDict


def _matches(name: str, patterns: Iterable[str]) -> bool:
    """Return True if `name` matches any of the patterns."""
    return any(re.fullmatch(pattern, name) for pattern in patterns)


def _should_train(name: str, patterns_unfreeze, patterns_freeze) -> tuple[bool, str | None]:
    """
    Determine whether a parameter should remain trainable and return a log action.
    """
    # Allow matching without the leading 'params.' prefix to mirror PyTorch names.
    alt_name = name[7:] if name.startswith('params.') else name

    if patterns_unfreeze:
        is_trainable = _matches(name, patterns_unfreeze) or _matches(
            alt_name, patterns_unfreeze
        )
        return is_trainable, 'Unfreezing' if is_trainable else 'Freezing'

    if patterns_freeze:
        should_freeze = _matches(name, patterns_freeze) or _matches(
            alt_name, patterns_freeze
        )
        return (not should_freeze), ('Freezing' if should_freeze else None)

    return True, None


def build_trainable_mask(args, params: FrozenDict, logger=None) -> FrozenDict | None:
    """
    Build a boolean PyTree mask indicating which parameters should receive updates.

    Returns
    -------
    FrozenDict | None
        Mask PyTree with True for trainable entries. ``None`` if no freeze flags were set.
    """
    patterns_unfreeze = tuple(getattr(args, 'unfreeze_params', []) or [])
    patterns_freeze = tuple(getattr(args, 'freeze_params', []) or [])

    if not patterns_unfreeze and not patterns_freeze:
        return None

    flat_params = traverse_util.flatten_dict(
        frozen_dict.unfreeze(params), sep='.'
    )  # type: ignore[arg-type]

    mask_entries: dict[str, bool] = {}
    for dotted_name in flat_params:
        trainable, action = _should_train(
            dotted_name, patterns_unfreeze, patterns_freeze
        )
        mask_entries[dotted_name] = trainable

        if logger is not None and action is not None:
            logger.log(1, f'{action} parameter: {dotted_name}')

    mask_tree = traverse_util.unflatten_dict(
        {tuple(name.split('.')): value for name, value in mask_entries.items()}
    )

    return frozen_dict.freeze(mask_tree)


__all__ = ['build_trainable_mask']
