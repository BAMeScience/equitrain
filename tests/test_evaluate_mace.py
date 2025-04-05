from equitrain import evaluate, get_args_parser_evaluate
from equitrain.utility_test import MaceWrapper


def test_mace_predict():
    args = get_args_parser_evaluate().parse_args()

    args.test_file = 'data/valid.h5'
    args.batch_size = 5
    args.model = MaceWrapper(args)
    args.verbose = 1

    evaluate(args)


if __name__ == '__main__':
    test_mace_predict()
