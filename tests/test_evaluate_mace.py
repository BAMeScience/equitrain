import csv
import json
from pathlib import Path

import pytest

pytest.importorskip('mace', reason='MACE is required for MACE integration tests.')

from equitrain import evaluate, get_args_parser_evaluate
from equitrain.utility_test import MaceWrapper
from equitrain.utility_test.mace_support import get_mace_model_path


def test_mace_predict(tmp_path):
    args = get_args_parser_evaluate().parse_args([])

    data_dir = Path(__file__).with_name('data')
    args.test_file = str(data_dir / 'valid.h5')
    args.batch_size = 5
    args.dtype = 'float32'
    args.output_dir = str(tmp_path / 'evaluate_mace')
    mace_model_path = get_mace_model_path()
    args.model = MaceWrapper(args, filename_model=mace_model_path)
    args.verbose = 1

    evaluate(args)

    output_dir = Path(args.output_dir)
    metrics = json.loads((output_dir / 'test_metrics.json').read_text())
    assert metrics['backend'] == 'torch'
    assert metrics['dataset'] == args.test_file
    assert metrics['loss_type'] == args.loss_type
    assert args.loss_type in metrics['metrics']

    with (output_dir / metrics['errors_file']).open(newline='') as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ['index', 'error']
    assert len(rows) > 1


if __name__ == '__main__':
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_mace_predict(Path(tmp_dir))
