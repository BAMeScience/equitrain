import torch

from equitrain.model_wrappers import *


def get_model(args, logger=None):

    if isinstance(args.model, torch.nn.Module):

        model = args.model

    else:

        model = torch.load(args.model)

    if args.load_checkpoint_model is not None:

        if logger is not None:
            logger.info(f'Loading model checkpoint {args.load_checkpoint_model}...')

        model.load_state_dict(torch.load(args.load_checkpoint_model))

    if args.model_wrapper == "mace":
        model = MaceWrapper(args, model)

    return model
