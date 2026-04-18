from __future__ import annotations

import argparse
from pathlib import Path

from equitrain.data.format_hdf5 import HDF5Dataset
from equitrain.data.format_lmdb import iter_lmdb_atoms


def _find_shards(src_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in src_dir.rglob('*.aselmdb')
        if path.is_file() and not path.name.endswith('-lock')
    )


def _convert_split(src_dir: Path, dst_file: Path) -> int:
    shards = _find_shards(src_dir)
    if not shards:
        raise FileNotFoundError(f'No .aselmdb shards found under {src_dir}')

    dst_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with HDF5Dataset(dst_file, mode='w') as dataset:
        for shard in shards:
            print(f'converting {shard}')
            for atoms in iter_lmdb_atoms(shard):
                dataset[count] = atoms
                count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert extracted OMAT24 ASE-LMDB shards into EquiTrain HDF5.'
    )
    parser.add_argument(
        '--train-dir',
        default='train',
        help='Directory containing extracted OMAT24 training shards.',
    )
    parser.add_argument(
        '--valid-dir',
        default='val',
        help='Directory containing extracted OMAT24 validation shards.',
    )
    parser.add_argument(
        '--output-dir',
        default='omat24',
        help='Output directory for train.h5 / valid.h5.',
    )
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Skip conversion of the training split.',
    )
    parser.add_argument(
        '--skip-valid',
        action='store_true',
        help='Skip conversion of the validation split.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    if not args.skip_train:
        train_count = _convert_split(
            Path(args.train_dir),
            output_dir / 'train.h5',
        )
        print(f'wrote {train_count} training structures to {output_dir / "train.h5"}')

    if not args.skip_valid:
        valid_count = _convert_split(
            Path(args.valid_dir),
            output_dir / 'valid.h5',
        )
        print(f'wrote {valid_count} validation structures to {output_dir / "valid.h5"}')


if __name__ == '__main__':
    main()
