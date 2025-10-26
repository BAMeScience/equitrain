from pathlib import Path

from equitrain import evaluate, get_args_parser_evaluate
from equitrain.utility_test import MaceWrapper


def test_mace_predict(mace_model_path):
    args = get_args_parser_evaluate().parse_args([])

    data_dir = Path(__file__).with_name('data')
    args.test_file = str(data_dir / 'valid.h5')
    args.batch_size = 5
    args.model = MaceWrapper(args, filename_model=mace_model_path)
    args.verbose = 1

    evaluate(args)


if __name__ == '__main__':
    test_mace_predict()
