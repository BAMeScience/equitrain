from pathlib import Path

import pytest

pytest.importorskip(
    'sevenn', reason='sevenn is required for SevenNet integration tests.'
)

from equitrain import get_args_parser_train, train
from equitrain.data.statistics_data import Statistics
from equitrain.utility_test import SevennetWrapper


def test_sevennet_atomic_numbers():
    args = get_args_parser_train().parse_args([])

    data_dir = Path(__file__).with_name('data')
    statistics_path = data_dir / 'statistics.json'
    statistics = Statistics.load(str(statistics_path))

    model = SevennetWrapper(
        args,
        filename_config=str(Path(__file__).with_name('test_train_sevennet.yaml')),
        filename_statistics=str(statistics_path),
    )

    assert model.atomic_numbers == statistics.atomic_numbers, (
        'atomic numbers do not match'
    )


def test_train_sevennet():
    args = get_args_parser_train().parse_args([])

    data_dir = Path(__file__).with_name('data')
    statistics_path = data_dir / 'statistics.json'
    args.train_file = str(data_dir / 'train.h5')
    args.valid_file = str(data_dir / 'valid.h5')
    args.test_file = str(data_dir / 'train.h5')
    args.output_dir = str(Path(__file__).with_name('test_train_sevennet'))
    args.model = SevennetWrapper(
        args,
        filename_config=str(Path(__file__).with_name('test_train_sevennet.yaml')),
        filename_statistics=str(statistics_path),
    )
    args.dtype = 'float32'

    args.epochs = 10
    args.batch_size = 2
    args.lr = 0.001
    args.verbose = 1
    args.tqdm = True

    train(args)


if __name__ == '__main__':
    test_sevennet_atomic_numbers()
    test_train_sevennet()
