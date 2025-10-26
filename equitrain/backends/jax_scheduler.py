from __future__ import annotations

import warnings

import optax


def scheduler_kwargs(args):
    return {
        'scheduler_name': getattr(args, 'scheduler', None),
        'learning_rate': getattr(args, 'lr', 1e-3),
        'gamma': getattr(args, 'gamma', 0.8),
        'step_size': getattr(args, 'step_size', 1),
        'min_lr': getattr(args, 'min_lr', 0.0),
    }


def create_scheduler(
    *,
    scheduler_name: str | None,
    learning_rate: float,
    gamma: float = 0.8,
    step_size: int = 1,
    min_lr: float = 0.0,
):
    name = (scheduler_name or 'constant').lower()

    if name in {'none', 'constant', ''}:
        return optax.constant_schedule(learning_rate)

    if name == 'exponential':
        return optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=max(step_size, 1),
            decay_rate=gamma,
            end_value=min_lr,
        )

    if name == 'step':
        boundaries_and_scales = {max(step_size, 1): gamma}
        return optax.piecewise_constant_schedule(learning_rate, boundaries_and_scales)

    if name == 'plateau':
        return optax.constant_schedule(learning_rate)

    warnings.warn(
        f'Unknown scheduler "{scheduler_name}" for JAX backend; using constant schedule.',
        stacklevel=2,
    )
    return optax.constant_schedule(learning_rate)


__all__ = ['scheduler_kwargs', 'create_scheduler']
