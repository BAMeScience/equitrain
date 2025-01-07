from equitrain import get_args_parser_train, train_fabric
from equitrain.utility_test import MaceWrapper


def test_train_mace_fabric():
    # r = 4.5  # ! unused

    args = get_args_parser_train().parse_args()

    args.train_file = 'data/train.h5'
    args.valid_file = 'data/valid.h5'
    args.statistics_file = 'data/statistics.json'
    args.output_dir = 'test_train_mace'
    args.model = MaceWrapper()

    args.epochs = 10
    args.batch_size = 64
    args.lr = 0.01

    train_fabric(args)
