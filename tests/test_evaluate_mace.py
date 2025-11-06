from pathlib import Path

import pytest

pytest.importorskip('mace', reason='MACE is required for MACE integration tests.')

from equitrain import evaluate, get_args_parser_evaluate
from equitrain.utility_test import MaceWrapper
from equitrain.utility_test.mace_support import get_mace_model_path


def test_mace_predict():
    args = get_args_parser_evaluate().parse_args([])

    data_dir = Path(__file__).with_name('data')
    args.test_file = str(data_dir / 'valid.h5')
    args.batch_size = 5
    args.dtype = 'float32'
    mace_model_path = get_mace_model_path()
    args.model = MaceWrapper(args, filename_model=mace_model_path)
    args.verbose = 1

    evaluate(args)


if __name__ == '__main__':
    test_mace_predict()
