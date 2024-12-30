
import torch

def add_weight_decay(model, weight_decay=1e-5, skip_list=[]):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (name.endswith(".bias") or name.endswith(".affine_weight")
            or name.endswith(".affine_bias") or name.endswith('.mean_shift')
            or 'bias.' in name
            or name in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay   , 'weight_decay': weight_decay}]


def optimizer_kwargs(args):

    kwargs = dict(
        optimizer_name = args.opt,
        lr             = args.lr,
        alpha          = args.alpha,
        weight_decay   = args.weight_decay,
        momentum       = args.momentum
    )

    return kwargs


def create_optimizer(args, model, filter_bias_and_bn=True) -> torch.optim.Optimizer:

    return create_optimizer_impl(
        model,
        **optimizer_kwargs(args=args),
        filter_bias_and_bn=filter_bias_and_bn,
    )


def create_optimizer_impl(
        model             : torch.nn.Module,
        optimizer_name    : str   = None,
        lr                : float = None,
        weight_decay      : float = None,
        alpha             : float = None,
        momentum          : float = None,
        filter_bias_and_bn: bool  = True,
        skip_list         : list  = [],
    )  -> torch.optim.Optimizer:

    opt_lower = optimizer_name.lower()

    if weight_decay and filter_bias_and_bn:
        parameters = add_weight_decay(model, weight_decay, skip_list)
        weight_decay = 0.0
    else:
        parameters = model.parameters()

    opt_args = dict(lr=lr, weight_decay=weight_decay)

    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        optimizer = torch.optim.SGD(parameters, momentum=momentum, nesterov=True)
    elif opt_lower == 'momentum':
        optimizer = torch.optim.SGD(parameters, momentum=momentum, nesterov=False)
    elif opt_lower == 'adam':
        optimizer = torch.optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = torch.optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = torch.optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = torch.optim.RMSprop(parameters, alpha=alpha, momentum=momentum, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer
