import torch


class SchedulerWrapper:
    def __init__(self, args, scheduler):
        """
        Wrapper for different LR schedulers.

        Args:
            scheduler: The learning rate scheduler (ExponentialLR or ReduceLROnPlateau).
            mode: "epoch" for schedulers like ExponentialLR, "metric" for ReduceLROnPlateau.
        """
        self.scheduler = scheduler
        self.mode = {'exponential': 'epoch', 'step': 'epoch', 'plateau': 'metric'}[
            args.scheduler
        ]

    def step(self, metric=None, epoch=None):
        """
        Steps the scheduler based on the mode.

        Args:
            metric: The monitored metric (required for ReduceLROnPlateau).
        """
        if self.mode == 'epoch':
            self.scheduler.step(epoch=epoch)
        elif self.mode == 'metric':
            if metric is None:
                raise ValueError('Metric is required for ReduceLROnPlateau')
            self.scheduler.step(metric)
        else:
            raise ValueError(f'Unsupported mode: {self.mode}')

    def get_last_lr(self):
        return self.scheduler.get_last_lr()


def scheduler_kwargs(args):
    kwargs = dict(
        scheduler_name=args.scheduler,
        gamma=args.gamma,
        min_lr=args.min_lr,
        eps=args.eps,
        step_size=args.step_size,
        plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        plateau_threshold=args.plateau_threshold,
        plateau_mode=args.plateau_mode,
    )
    return kwargs


def create_scheduler(
    args,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler:
    return create_scheduler_impl(
        optimizer=optimizer,
        **scheduler_kwargs(args),
    )


def create_scheduler_impl(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = None,
    gamma: float = None,
    min_lr: float = None,
    eps: float = None,
    step_size: int = None,
    plateau_factor: float = None,
    plateau_patience: int = None,
    plateau_threshold: float = None,
    plateau_mode: str = None,
) -> torch.optim.lr_scheduler.LRScheduler:
    lr_scheduler = None

    if scheduler_name == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    elif scheduler_name == 'exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma,
        )

    elif scheduler_name == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=plateau_mode,
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=plateau_threshold,
            threshold_mode='rel',
            min_lr=min_lr,
            eps=eps,
        )

    return lr_scheduler
