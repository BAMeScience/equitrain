import torch

from equitrain.model_wrappers import *


# TODO: Use arguments compute_force, compute_stress
def get_model(r_max, args, compute_force=True, compute_stress=True, logger=None):

    if isinstance(args.model, torch.nn.Module):

        model = args.model

    else:

        model = torch.load(args.model)

    if args.load_checkpoint_model is not None:

        if logger is not None:
            logger.info(f'Loading model checkpoint {args.load_checkpoint_model}...')

        model.load_state_dict(torch.load(args.load_checkpoint_model))

    if args.model_wrapper == "mace":
        model = MaceWrapper(model)

    return model
