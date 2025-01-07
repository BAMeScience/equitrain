# %%
from equitrain import get_args_parser_preprocess, preprocess


# %%
def main():
    args = get_args_parser_preprocess().parse_args()

    args.train_file = 'alexandria_mptraj_train.xyz'
    args.valid_file = 'alexandria_mptraj_valid.xyz'
    args.statistics_file = 'statistics.json'
    args.output_dir = 'alexandria_mptraj'
    args.compute_statistics = True
    args.atomic_energies = 'average'
    args.r_max = 6.0

    preprocess(args)


# %%
if __name__ == '__main__':
    main()
