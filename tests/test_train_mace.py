from pathlib import Path

from equitrain import get_args_parser_train, train
from equitrain.utility_test import MaceWrapper


def test_train_mace(tmp_path):
    args = get_args_parser_train().parse_args([])

    data_dir = Path(__file__).with_name('data')
    args.train_file = str(data_dir / 'train.h5')
    args.valid_file = str(data_dir / 'valid.h5')
    args.test_file = str(data_dir / 'train.h5')
    output_dir = tmp_path / 'train_mace'
    args.output_dir = str(output_dir)
    args.model = MaceWrapper(args)

    args.epochs = 10
    args.batch_size = 2
    args.lr = 0.001
    args.verbose = 1
    args.tqdm = True

    train(args)


if __name__ == '__main__':
    test_train_mace()
