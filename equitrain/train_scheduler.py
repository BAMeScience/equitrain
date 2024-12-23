
import torch

def scheduler_kwargs(args):

    kwargs = dict(
        scheduler_name    = args.scheduler,
        gamma             = args.epochs,
        min_lr            = args.min_lr,
        eps               = args.eps,
        plateau_factor    = args.plateau_factor,
        plateau_patience  = args.plateau_patience,
        plateau_threshold = args.plateau_threshold,
        plateau_mode      = args.plateau_mode,
    )
    return kwargs


def create_scheduler(
        args,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:

    return create_scheduler_impl(
        optimizer = optimizer,
        **scheduler_kwargs(args),
    )


def create_scheduler_impl(
        optimizer         : torch.optim.Optimizer,
        scheduler_name    : str   = None,
        gamma             : float = None,
        min_lr            : float = None,
        eps               : float = None,
        plateau_factor    : float = None,
        plateau_patience  : int   = None,
        plateau_threshold : float = None,
        plateau_mode      : str   = None,
        verbose           : int   = None,
    ) -> torch.optim.lr_scheduler.LRScheduler:

    lr_scheduler = None

    if scheduler_name == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            step_size      = step_size,
            gamma          = gamma,
        )

    elif scheduler_name == 'exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma          = gamma,
        )

    elif scheduler_name == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode           = plateau_mode,
            factor         = plateau_factor,
            patience       = plateau_patience,
            threshold      = plateau_threshold,
            threshold_mode = 'rel',
            min_lr         = min_lr,
            eps            = eps,
        )

    return lr_scheduler
