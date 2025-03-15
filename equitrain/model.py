import torch

from equitrain.model_wrappers import MaceWrapper, SevennetWrapper


def get_model(args, logger=None):
    if isinstance(args.model, torch.nn.Module):
        model = args.model

    else:
        # TODO: Check if file exists. Through a meaningful error if not
        model = torch.load(args.model, weights_only=False)

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

    return model
