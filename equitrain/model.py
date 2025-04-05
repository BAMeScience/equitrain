import torch

from equitrain.model_wrappers import AbstractWrapper, MaceWrapper, SevennetWrapper


def get_model(args, logger=None):
    if isinstance(args.model, torch.nn.Module):
        model = args.model

    else:
        # TODO: Check if file exists. Through a meaningful error if not
        model = torch.load(args.model, weights_only=False)

    # Apply model wrapper
    if not isinstance(model, AbstractWrapper):
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

    # Overwrite model parameters
    if hasattr(args, 'r_max') and args.r_max is not None:
        if logger is not None:
            logger.log(1, f'Overwriting r_max model parameter with r_max={args.r_max}')
        model.r_max = args.r_max

    return model
