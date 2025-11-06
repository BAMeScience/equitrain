from __future__ import annotations

import optax


def optimizer_kwargs(args):
    return {
        'optimizer_name': getattr(args, 'opt', 'adamw'),
        'learning_rate': getattr(args, 'lr', 1e-3),
        'weight_decay': getattr(args, 'weight_decay', 0.0) or 0.0,
        'momentum': getattr(args, 'momentum', 0.0) or 0.0,
        'alpha': getattr(args, 'alpha', 0.99) or 0.99,
    }


def create_optimizer(
    *,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    alpha: float = 0.99,
    learning_rate_schedule=None,
    mask=None,
):
    schedule = learning_rate_schedule or optax.constant_schedule(learning_rate)
    name = (optimizer_name or 'adamw').lower()
    weight_decay = float(weight_decay or 0.0)

    if name == 'adamw':
        base = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    elif name == 'adam':
        base = optax.adam(learning_rate=schedule)
        if weight_decay:
            base = optax.chain(optax.add_decayed_weights(weight_decay), base)
    elif name in {'sgd', 'nesterov'}:
        base = optax.sgd(
            learning_rate=schedule,
            momentum=momentum,
            nesterov=True if name == 'nesterov' else False,
        )
        if weight_decay:
            base = optax.chain(optax.add_decayed_weights(weight_decay), base)
    elif name == 'momentum':
        base = optax.sgd(
            learning_rate=schedule,
            momentum=momentum,
            nesterov=False,
        )
        if weight_decay:
            base = optax.chain(optax.add_decayed_weights(weight_decay), base)
    elif name == 'rmsprop':
        base = optax.rmsprop(
            learning_rate=schedule,
            decay=alpha,
            momentum=momentum,
        )
        if weight_decay:
            base = optax.chain(optax.add_decayed_weights(weight_decay), base)
    else:
        raise ValueError(f'Unsupported optimizer for JAX backend: {optimizer_name}')

    if mask is not None:
        base = optax.masked(base, mask)

    return base


__all__ = ['optimizer_kwargs', 'create_optimizer']
