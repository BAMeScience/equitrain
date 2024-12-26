# %%
from equitrain import get_args_parser_preprocess
from equitrain import preprocess

# %%
def main():

    args = get_args_parser_preprocess().parse_args()
    
    args.train_file         = 'mptraj_train.xyz'
    args.valid_file         = 'mptraj_valid.xyz'
    args.statistics_file    = 'statistics.json'
    args.output_dir         = 'mptraj'
    args.compute_statistics = True
    args.E0s                = "average"
    args.r_max              = 6.0

    preprocess(args)


# %%
if __name__ == "__main__":
    main()
