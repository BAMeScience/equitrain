# %%
from equitrain import get_args_parser_preprocess, preprocess


# %%
def main():
    args = get_args_parser_preprocess().parse_args()

    args.train_file = 'alexandria_train.xyz'
    args.valid_file = 'alexandria_valid.xyz'
    args.output_dir = 'alexandria'
    args.compute_statistics = True
    args.atomic_energies = 'average'
    args.r_max = 6.0

    preprocess(args)


# %%
if __name__ == '__main__':
    main()
