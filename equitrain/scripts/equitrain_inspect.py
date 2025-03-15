import sys

import torch
from tabulate import tabulate

from equitrain import check_args_complete, get_args_parser_inspect
from equitrain.model import get_model


# %%
def _inspect(args):
    model = get_model(args)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Collect parameters
    param_list = [
        (name, tuple(param.shape), param.requires_grad)
        for name, param in model.named_parameters()
    ]

    print(f'Total number of parameters: {n_parameters}\n')
    print(tabulate(param_list, headers=['Name', 'Shape', 'Requires Grad']))


# %%
def inspect(args):
    check_args_complete(args, 'inspect')

    if args.model is None:
        raise ValueError('--model is a required argument')

    _inspect(args)


# %%
def main():
    parser = get_args_parser_inspect()

    try:
        inspect(parser.parse_args())

    except ValueError as v:
        print(v, file=sys.stderr)
        sys.exit(1)


# %%
if __name__ == '__main__':
    main()
