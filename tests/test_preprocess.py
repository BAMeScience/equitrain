# %%

import pytest

from equitrain import get_args_parser_preprocess
from equitrain import preprocess


# %%
def test_preprocess():

    args = get_args_parser_preprocess().parse_args()
    
    args.train_file         = 'data.xyz'
    args.valid_file         = 'data.xyz'
    args.output_dir         = 'test_preprocess/'
    args.compute_statistics = True
    args.atomic_energies    = "average"
    args.r_max              = 4.5

    preprocess(args)


# %%
if __name__ == "__main__":
    test_preprocess()
