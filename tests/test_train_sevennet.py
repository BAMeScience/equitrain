from equitrain import get_args_parser_train, train
from equitrain.data import Statistics
from equitrain.utility_test import SevennetWrapper


def test_sevennet_atomic_numbers():
    args = get_args_parser_train().parse_args()

    statistics = Statistics.load('data/statistics.json')

    model = SevennetWrapper(
        args,
        filename_config='test_train_sevennet.yaml',
        filename_statistics='data/statistics.json',
    )

    assert model.atomic_numbers == statistics.atomic_numbers, (
        'atomic numbers do not match'
    )


def test_train_sevennet():
    args = get_args_parser_train().parse_args()

    args.train_file = 'data/train.h5'
    args.valid_file = 'data/valid.h5'
    args.test_file = 'data/train.h5'
    args.output_dir = 'test_train_sevennet'
    args.model = SevennetWrapper(
        args,
        filename_config='test_train_sevennet.yaml',
        filename_statistics='data/statistics.json',
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
