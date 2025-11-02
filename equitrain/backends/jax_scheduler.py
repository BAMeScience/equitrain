from __future__ import annotations

from dataclasses import dataclass


def scheduler_kwargs(args):
    return {
        'scheduler_name': getattr(args, 'scheduler', None),
        'gamma': getattr(args, 'gamma', 0.8),
        'step_size': getattr(args, 'step_size', 1),
        'min_lr': getattr(args, 'min_lr', 0.0),
        'plateau_mode': getattr(args, 'plateau_mode', 'min'),
        'plateau_factor': getattr(args, 'plateau_factor', 0.1),
        'plateau_patience': getattr(args, 'plateau_patience', 10),
        'plateau_threshold': getattr(args, 'plateau_threshold', 0.0001),
        'plateau_threshold_mode': getattr(args, 'plateau_threshold_mode', 'rel'),
        'plateau_eps': getattr(args, 'plateau_eps', 1e-8),
        'monitor': getattr(args, 'scheduler_monitor', 'val'),
        'start_epoch': getattr(args, 'epochs_start', 1),
    }


@dataclass
class _SchedulerBase:
    name: str
    monitor: str
    current_lr: float
    min_lr: float
    start_epoch: int

    def register_initial_metric(self, metric, epoch: int) -> None:
        """Allow controllers to update their internal state before training begins."""

    def update_after_epoch(self, *, metric, epoch: int) -> bool:
        """Update the learning rate after an epoch. Returns True if lr changed."""
        return False


@dataclass
class _ConstantScheduler(_SchedulerBase):
    pass


@dataclass
class _StepScheduler(_SchedulerBase):
    gamma: float
    step_size: int

    def update_after_epoch(self, *, metric, epoch: int) -> bool:
        if self.step_size <= 0:
            return False
        steps_completed = epoch - self.start_epoch + 1
        if steps_completed <= 0:
            return False
        if steps_completed % self.step_size != 0:
            return False
        new_lr = max(self.current_lr * self.gamma, self.min_lr)
        if new_lr < self.current_lr:
            self.current_lr = new_lr
            return True
        return False


@dataclass
class _ExponentialScheduler(_SchedulerBase):
    gamma: float

    def update_after_epoch(self, *, metric, epoch: int) -> bool:
        if epoch < self.start_epoch:
            return False
        new_lr = max(self.current_lr * self.gamma, self.min_lr)
        if new_lr < self.current_lr:
            self.current_lr = new_lr
            return True
        return False


@dataclass
class _PlateauScheduler(_SchedulerBase):
    factor: float
    patience: int
    threshold: float
    threshold_mode: str
    mode: str
    eps: float
    best: float | None = None
    num_bad_epochs: int = 0

    def register_initial_metric(self, metric, epoch: int) -> None:
        if metric is not None and (
            self.best is None or self._is_better(metric, self.best)
        ):
            self.best = float(metric)
            self.num_bad_epochs = 0

    def update_after_epoch(self, *, metric, epoch: int) -> bool:
        if metric is None:
            return False
        if self.best is None:
            self.best = float(metric)
            self.num_bad_epochs = 0
            return False
        if self._is_better(metric, self.best):
            self.best = float(metric)
            self.num_bad_epochs = 0
            return False
        self.num_bad_epochs += 1
        if self.num_bad_epochs <= self.patience:
            return False
        self.num_bad_epochs = 0
        new_lr = max(self.current_lr * self.factor, self.min_lr)
        if new_lr < self.current_lr - self.eps:
            self.current_lr = new_lr
            return True
        return False

    def _is_better(self, metric: float, best: float) -> bool:
        if self.mode == 'max':
            comparison = metric > best
        else:
            comparison = metric < best
        if not comparison:
            return False
        if self.threshold_mode == 'rel':
            if self.mode == 'max':
                return metric > best * (1.0 + self.threshold)
            return metric < best * (1.0 - self.threshold)
        if self.mode == 'max':
            return metric > best + self.threshold
        return metric < best - self.threshold


def create_scheduler_controller(args, initial_lr: float):
    cfg = scheduler_kwargs(args)
    name = (cfg['scheduler_name'] or 'constant').lower()
    monitor = (cfg['monitor'] or 'val').lower()
    if name in {'none', 'constant', ''}:
        return _ConstantScheduler(
            name='constant',
            monitor=monitor,
            current_lr=float(initial_lr),
            min_lr=float(cfg['min_lr']),
            start_epoch=int(cfg['start_epoch']),
        )
    if name == 'step':
        return _StepScheduler(
            name='step',
            monitor=monitor,
            current_lr=float(initial_lr),
            min_lr=float(cfg['min_lr']),
            start_epoch=int(cfg['start_epoch']),
            gamma=float(cfg['gamma']),
            step_size=max(int(cfg['step_size']), 1),
        )
    if name == 'exponential':
        return _ExponentialScheduler(
            name='exponential',
            monitor=monitor,
            current_lr=float(initial_lr),
            min_lr=float(cfg['min_lr']),
            start_epoch=int(cfg['start_epoch']),
            gamma=float(cfg['gamma']),
        )
    if name == 'plateau':
        return _PlateauScheduler(
            name='plateau',
            monitor=monitor,
            current_lr=float(initial_lr),
            min_lr=float(cfg['min_lr']),
            start_epoch=int(cfg['start_epoch']),
            factor=float(cfg['plateau_factor']),
            patience=max(int(cfg['plateau_patience']), 0),
            threshold=float(cfg['plateau_threshold']),
            threshold_mode=str(cfg['plateau_threshold_mode']).lower(),
            mode=str(cfg['plateau_mode']).lower(),
            eps=float(cfg['plateau_eps']),
        )
    # fallback to constant
    return _ConstantScheduler(
        name='constant',
        monitor=monitor,
        current_lr=float(initial_lr),
        min_lr=float(cfg['min_lr']),
        start_epoch=int(cfg['start_epoch']),
    )


__all__ = ['scheduler_kwargs', 'create_scheduler_controller']
