"""JAX utilities for freezing/unfreezing parameters via regex patterns."""

from __future__ import annotations

import re
from collections.abc import Iterable

from flax import traverse_util
from flax.core import frozen_dict
from flax.core.frozen_dict import FrozenDict


def _matches(name: str, patterns: Iterable[str]) -> bool:
    """Return True if `name` matches any of the patterns."""
    return any(re.fullmatch(pattern, name) for pattern in patterns)


def _path_to_name(path: tuple[object, ...]) -> str:
    return '.'.join(str(part) for part in path)


def _should_train(
    name: str,
    patterns_unfreeze,
    patterns_freeze,
    default_trainable: bool,
) -> tuple[bool, str | None]:
    """
    Determine whether a parameter should remain trainable and return a log action.
    """
    candidate_names = {name}
    if name.startswith('params.'):
        candidate_names.add(name[7:])

    if patterns_unfreeze:
        is_trainable = any(
            _matches(candidate, patterns_unfreeze) for candidate in candidate_names
        )
        return is_trainable, 'Unfreezing' if is_trainable else 'Freezing'

    if patterns_freeze:
        should_freeze = any(
            _matches(candidate, patterns_freeze) for candidate in candidate_names
        )
        return (not should_freeze), ('Freezing' if should_freeze else None)

    return default_trainable, None


def build_trainable_mask(
    args,
    params: FrozenDict,
    logger=None,
    *,
    default_trainable: bool = True,
) -> FrozenDict | dict | None:
    """
    Build a boolean PyTree mask indicating which parameters should receive updates.

    Returns
    -------
    FrozenDict | dict | None
        Mask PyTree with True for trainable entries, matching the input container
        type. ``None`` if no freeze flags were set.
    """
    patterns_unfreeze = tuple(getattr(args, 'unfreeze_params', []) or [])
    patterns_freeze = tuple(getattr(args, 'freeze_params', []) or [])

    if not patterns_unfreeze and not patterns_freeze:
        if default_trainable:
            return None

    flat_params = traverse_util.flatten_dict(frozen_dict.unfreeze(params))  # type: ignore[arg-type]

    mask_entries: dict[tuple[object, ...], bool] = {}
    for path in flat_params:
        dotted_name = _path_to_name(path)
        trainable, action = _should_train(
            dotted_name,
            patterns_unfreeze,
            patterns_freeze,
            default_trainable,
        )
        mask_entries[path] = trainable

        if logger is not None and action is not None:
            logger.log(1, f'{action} parameter: {dotted_name}')

    mask_tree = traverse_util.unflatten_dict(mask_entries)

    if isinstance(params, FrozenDict):
        return frozen_dict.freeze(mask_tree)
    return mask_tree


__all__ = ['build_trainable_mask']
