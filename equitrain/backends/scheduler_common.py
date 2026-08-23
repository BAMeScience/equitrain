from __future__ import annotations


def scheduler_kwargs(args):
    """
    Collect scheduler-related hyperparameters that are backend agnostic.
    """

    decay_rate = getattr(args, 'decay_rate', None)
    gamma = getattr(args, 'gamma', 0.8) if decay_rate is None else decay_rate

    return {
        'scheduler_name': getattr(args, 'scheduler', None),
        'gamma': gamma,
        'min_lr': getattr(args, 'min_lr', 0.0),
        'step_size': getattr(args, 'step_size', 5),
        'plateau_mode': getattr(args, 'plateau_mode', 'min'),
        'plateau_factor': getattr(args, 'plateau_factor', 0.5),
        'plateau_patience': getattr(args, 'plateau_patience', 2),
        'plateau_threshold': getattr(args, 'plateau_threshold', 0.0001),
        'plateau_threshold_mode': getattr(args, 'plateau_threshold_mode', 'rel'),
        'plateau_eps': getattr(args, 'plateau_eps', 1e-12),
    }


__all__ = ['scheduler_kwargs']
