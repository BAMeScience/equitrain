from equitrain import get_args_parser_train, train
from equitrain.utility_test import MaceWrapper


def test_train_mace():
    # r = 4.5  # ! unused

    args = get_args_parser_train().parse_args()

    args.train_file = 'data/train.h5'
    args.valid_file = 'data/valid.h5'
    args.test_file = 'data/train.h5'
    args.statistics_file = 'data/statistics.json'
    args.output_dir = 'test_train_mace'
    args.model = MaceWrapper(args)

    args.epochs = 10
    args.batch_size = 2
    args.lr = 0.001
    args.verbose = 2
    args.tqdm = True

    train(args)
