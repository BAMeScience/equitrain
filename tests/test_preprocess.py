from pathlib import Path

from equitrain import get_args_parser_preprocess, preprocess


def test_preprocess(tmp_path):
    args = get_args_parser_preprocess().parse_args([])

    data_xyz = Path(__file__).with_name('data.xyz')
    args.train_file = str(data_xyz)
    args.valid_file = str(data_xyz)
    output_dir = tmp_path / 'preprocess'
    args.output_dir = str(output_dir)
    args.compute_statistics = True
    args.atomic_energies = 'average'
    args.r_max = 4.5
    args.verbose = 1

    preprocess(args)


if __name__ == '__main__':
    test_preprocess()
