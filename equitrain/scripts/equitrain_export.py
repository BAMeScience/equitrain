import sys

import torch

from equitrain import check_args_complete, get_args_parser_export
from equitrain.argparser import ArgsFormatter
from equitrain.checkpoint import load_checkpoint, save_checkpoint
from equitrain.logger import FileLogger
from equitrain.model import get_model
from equitrain.model_wrappers import AbstractWrapper


# %%
def _export(args):
    logger = FileLogger(
        log_to_file=True,
        enable_logging=True,
        verbosity=args.verbose,
    )
    logger.log(1, ArgsFormatter(args))

    model = get_model(args)

    # Import model, optimizer, lr_scheduler from checkpoint if possible
    if not load_checkpoint(args, model, logger=logger):
        logger.log(1, 'No checkpoint found, using initial model')

    if isinstance(model, AbstractWrapper):
        model = model.model

    model = model.cpu()

    torch.save(model, args.model_export)


# %%
def export(args):
    check_args_complete(args, 'export')

    if args.model is None:
        raise ValueError('--model is a required argument')
    if args.model_export is None:
        raise ValueError('--model-export is a required argument')

    _export(args)


# %%
def main():
    parser = get_args_parser_export()

    try:
        export(parser.parse_args())

    except ValueError as v:
        print(v, file=sys.stderr)
        sys.exit(1)


# %%
if __name__ == '__main__':
    main()
