import pytest

from equitrain import get_args_parser_train
from equitrain import train

from equitrain.utility_test import MaceWrapper

# %%

def test_train_mace():

    r = 4.5

    args = get_args_parser_train().parse_args()

    args.train_file      = f'data/train.h5'
    args.valid_file      = f'data/valid.h5'
    args.statistics_file = f'data/statistics.json'
    args.output_dir      = 'test_train_mace'
    args.model           = MaceWrapper()

    args.epochs     = 10
    args.batch_size = 64
    args.lr         = 0.01
    args.verbose    = 2

    train(args)

# %%
if __name__ == "__main__":
    test_train_mace()
