from __future__ import annotations

import torch


class SchedulerWrapper:
    def __init__(self, args, scheduler):
        self.scheduler = scheduler
        self.mode = {'exponential': 'epoch', 'step': 'epoch', 'plateau': 'metric'}[
            args.scheduler
        ]

    def step(self, metric=None, epoch=None):
        if self.mode == 'epoch':
            if epoch is not None:
                self.scheduler.step(epoch=epoch)
        elif self.mode == 'metric':
            if metric is not None:
                self.scheduler.step(metric)
        else:
            raise ValueError(f'Unsupported mode: {self.mode}')

    def get_last_lr(self):
        return self.scheduler.get_last_lr()


def scheduler_kwargs(args):
    return dict(
        scheduler_name=args.scheduler,
        gamma=args.gamma,
        min_lr=args.min_lr,
        step_size=args.step_size,
        plateau_mode=args.plateau_mode,
        plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        plateau_threshold=args.plateau_threshold,
        plateau_threshold_mode=args.plateau_threshold_mode,
        plateau_eps=args.plateau_eps,
    )


def create_scheduler(args, optimizer: torch.optim.Optimizer):
    return create_scheduler_impl(optimizer=optimizer, **scheduler_kwargs(args))


def create_scheduler_impl(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = None,
    gamma: float = None,
    min_lr: float = None,
    step_size: int = None,
    plateau_factor: float = None,
    plateau_patience: int = None,
    plateau_threshold: float = None,
    plateau_mode: str = None,
    plateau_threshold_mode: str = None,
    plateau_eps: float = None,
):
    if scheduler_name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    if scheduler_name == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma,
        )
    if scheduler_name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=plateau_mode,
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=plateau_threshold,
            threshold_mode=plateau_threshold_mode,
            eps=plateau_eps,
            min_lr=min_lr,
        )
    raise ValueError(f'Unsupported scheduler: {scheduler_name}')


__all__ = [
    'SchedulerWrapper',
    'scheduler_kwargs',
    'create_scheduler',
    'create_scheduler_impl',
]
