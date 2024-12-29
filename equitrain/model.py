import torch

from equitrain.model_wrappers import *


def get_model(args, logger=None):

    if isinstance(args.model, torch.nn.Module):

        model = args.model

    else:

        model = torch.load(args.model)

    if args.model_checkpoint is not None:

        if logger is not None:
            logger.info(f'Loading model checkpoint {args.model_checkpoint}...')

        model.load_state_dict(torch.load(args.model_checkpoint))

    if args.model_wrapper == "mace":
        model = MaceWrapper(args, model)

    return model
