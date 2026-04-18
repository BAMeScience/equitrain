from __future__ import annotations

from pathlib import Path

from equitrain import get_args_parser_preprocess, preprocess


def main():
    args = get_args_parser_preprocess().parse_args()

    output_dir = Path('omat24')
    args.train_file = str(output_dir / 'train.h5')
    args.valid_file = str(output_dir / 'valid.h5')
    args.output_dir = str(output_dir)
    args.compute_statistics = True
    args.atomic_energies = 'average'
    args.r_max = 6.0

    preprocess(args)


if __name__ == '__main__':
    main()
