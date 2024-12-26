# %%

import pytest

from equitrain import get_args_parser_preprocess
from equitrain import preprocess


# %%
def test_preprocess():

    args = get_args_parser_preprocess().parse_args()
    
    args.train_file         = 'alexandria_train.xyz'
    args.valid_file         = 'alexandria_valid.xyz'
    args.statistics_file    = 'statistics.json'
    args.output_dir         = 'alexandria'
    args.compute_statistics = True
    args.E0s                = "average"
    args.r_max              = 6.0

    preprocess(args)


# %%
if __name__ == "__main__":
    test_preprocess()
