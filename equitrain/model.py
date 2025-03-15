import re

import torch

from equitrain.model_wrappers import MaceWrapper, SevennetWrapper


def model_freeze_params(args, model, logger=None):
    # Freeze all parameters by default if unfreeze_params is specified
    if hasattr(args, 'unfreeze_params') and args.unfreeze_params:
        for name, param in model.named_parameters():
            param.requires_grad = any(
                re.fullmatch(pattern, name) for pattern in args.unfreeze_params
            )
            if not param.requires_grad and logger is not None:
                logger.log(1, f'Freezing parameter: {name}')
            elif param.requires_grad and logger is not None:
                logger.log(1, f'Unfreezing parameter: {name}')

    # Freeze specified parameters using regex
    elif hasattr(args, 'freeze_params') and args.freeze_params:
        for name, param in model.named_parameters():
            if any(re.fullmatch(pattern, name) for pattern in args.freeze_params):
                param.requires_grad = False
                if logger is not None:
                    logger.log(1, f'Freezing parameter: {name}')


def get_model(args, logger=None):
    if isinstance(args.model, torch.nn.Module):
        model = args.model

    else:
        # TODO: Check if file exists. Through a meaningful error if not
        model = torch.load(args.model, weights_only=False)

    # Set attributes that might be required for the wrapper
    if not hasattr(args, 'energy_weight'):
        setattr(args, 'energy_weight', 0.0)
    if not hasattr(args, 'forces_weight'):
        setattr(args, 'forces_weight', 0.0)
    if not hasattr(args, 'stress_weight'):
        setattr(args, 'stress_weight', 0.0)

    if args.model_wrapper == 'mace':
        model = MaceWrapper(args, model)
    if args.model_wrapper == 'sevennet':
        model = SevennetWrapper(args, model)

    if (
        hasattr(args, 'load_checkpoint_model')
        and args.load_checkpoint_model is not None
    ):
        if logger is not None:
            logger.log(1, f'Loading model checkpoint {args.load_checkpoint_model}...')

        model.load_state_dict(torch.load(args.load_checkpoint_model, weights_only=True))

    model_freeze_params(args, model, logger=logger)

    return model
