#! /usr/bin/env python
# %%
from equitrain import get_args_parser_preprocess
from equitrain import preprocess
# %%
def main():
    args = get_args_parser_preprocess().parse_args()
    
    args.train_file      = 'data.xyz'
    args.valid_file      = 'data.xyz'
    args.statistics_file = 'output_fix/statistics2.json'
    args.atomic_numbers     = '[3, 8, 9, 11, 12, 13, 15, 17, 21, 22, 24, 25, 27, 28, 29, 31, 41, 44, 47, 79]'
    args.output_dir         = 'output_fix/'
    args.compute_statistics = True
    args.E0s                = "average"
    args.r_max              = 4.5
    print(args)
    preprocess(args)
# %%
if __name__ == "__main__":
    main()
