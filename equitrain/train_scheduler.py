"""Compatibility shims for torch learning rate schedulers."""

from __future__ import annotations


def SchedulerWrapper(*args, **kwargs):  # noqa: N802
    from equitrain.backends.torch_scheduler import SchedulerWrapper as _Wrapper

    return _Wrapper(*args, **kwargs)


def scheduler_kwargs(*args, **kwargs):
    from equitrain.backends.torch_scheduler import scheduler_kwargs as _impl

    return _impl(*args, **kwargs)


def create_scheduler(*args, **kwargs):
    from equitrain.backends.torch_scheduler import create_scheduler as _impl

    return _impl(*args, **kwargs)


def create_scheduler_impl(*args, **kwargs):
    from equitrain.backends.torch_scheduler import create_scheduler_impl as _impl

    return _impl(*args, **kwargs)


__all__ = [
    'SchedulerWrapper',
    'scheduler_kwargs',
    'create_scheduler',
    'create_scheduler_impl',
]
