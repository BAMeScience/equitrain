from __future__ import annotations

import math


def normalize_ratio(
    value: float | int | None,
    *,
    name: str,
    allow_zero: bool,
) -> float | None:
    if value is None:
        return None

    ratio = float(value)
    if ratio > 1.0:
        if ratio > 100.0:
            raise ValueError(
                f'{name} must be expressed as a fraction in [0, 1] or percentage in [0, 100].'
            )
        ratio /= 100.0

    if allow_zero:
        if not 0.0 <= ratio < 1.0:
            raise ValueError(f'{name} must be in [0, 1) or [0, 100).')
    else:
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f'{name} must be in (0, 1] or (0, 100].')
    return ratio


def resolve_retained_fraction(
    *,
    rank_fraction: float | int | None,
    rank_reduction: float | int | None,
    default: float = 0.25,
) -> float:
    if rank_fraction is not None and rank_reduction is not None:
        raise ValueError('Specify at most one of `rank_fraction` or `rank_reduction`.')

    fraction = normalize_ratio(
        rank_fraction,
        name='rank_fraction',
        allow_zero=False,
    )
    if fraction is not None:
        return fraction

    reduction = normalize_ratio(
        rank_reduction,
        name='rank_reduction',
        allow_zero=True,
    )
    if reduction is not None:
        retained = 1.0 - reduction
        if retained <= 0.0:
            raise ValueError('rank_reduction must leave a positive retained rank.')
        return retained

    return float(default)


def resolve_rank(
    effective_rank: int,
    *,
    retained_fraction: float,
    min_rank: int,
) -> int:
    if effective_rank <= 0:
        raise ValueError('effective_rank must be positive.')
    if min_rank <= 0:
        raise ValueError('min_rank must be positive.')
    rank = int(math.ceil(retained_fraction * effective_rank))
    return max(min_rank, min(rank, effective_rank))


def effective_matrix_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    out_dim = int(shape[0])
    in_dim = int(math.prod(shape[1:]))
    return out_dim, in_dim


__all__ = [
    'effective_matrix_shape',
    'normalize_ratio',
    'resolve_rank',
    'resolve_retained_fraction',
]
