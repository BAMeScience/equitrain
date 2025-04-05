import re


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
