from __future__ import annotations

import torch


def update_weight_decay(args, logger, optimizer):
    if args.weight_decay is None:
        return

    for param_group in optimizer.param_groups:
        if (
            param_group.get('name', 'n/a') == 'decay'
            or param_group['weight_decay'] > 0.0
        ):
            param_group['weight_decay'] = args.weight_decay

    logger.log(1, f'Using weight_decay = {args.weight_decay}')


def add_weight_decay(model, weight_decay, skip_list=None):
    if skip_list is None:
        skip_list = []

    decay = []
    nocay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            name.endswith('.bias')
            or name.endswith('.affine_weight')
            or name.endswith('.affine_bias')
            or name.endswith('.mean_shift')
            or 'bias.' in name
            or name in skip_list
        ):
            nocay.append(param)
        else:
            decay.append(param)

    return [
        {'params': nocay, 'name': 'nocay', 'weight_decay': 0.0},
        {'params': decay, 'name': 'decay', 'weight_decay': weight_decay},
    ]


def optimizer_kwargs(args):
    return dict(
        optimizer_name=args.opt,
        lr=args.lr,
        alpha=args.alpha,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )


def create_optimizer(args, model, filter_bias_and_bn=True) -> torch.optim.Optimizer:
    return create_optimizer_impl(
        model,
        **optimizer_kwargs(args=args),
        filter_bias_and_bn=filter_bias_and_bn,
    )


def create_optimizer_impl(
    model: torch.nn.Module,
    optimizer_name: str = None,
    lr: float = None,
    weight_decay: float = None,
    alpha: float = None,
    momentum: float = None,
    filter_bias_and_bn: bool = True,
    skip_list: list = None,
) -> torch.optim.Optimizer:
    if skip_list is None:
        skip_list = []

    opt_lower = optimizer_name.lower()

    if weight_decay is None:
        weight_decay = 0.0

    if filter_bias_and_bn:
        parameters = add_weight_decay(model, weight_decay, skip_list)
        weight_decay = 0.0
    else:
        parameters = model.parameters()

    opt_args = dict(lr=lr, weight_decay=weight_decay)

    if opt_lower in {'sgd', 'nesterov'}:
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            nesterov=True,
            weight_decay=weight_decay,
        )
    elif opt_lower == 'momentum':
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            nesterov=False,
            weight_decay=weight_decay,
        )
    elif opt_lower == 'adam':
        optimizer = torch.optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = torch.optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = torch.optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            parameters, alpha=alpha, momentum=momentum, **opt_args
        )
    else:
        raise ValueError('Invalid optimizer')

    return optimizer


__all__ = [
    'update_weight_decay',
    'add_weight_decay',
    'optimizer_kwargs',
    'create_optimizer',
    'create_optimizer_impl',
]
