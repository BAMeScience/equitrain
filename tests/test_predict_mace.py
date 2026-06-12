import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('mace', reason='MACE is required for MACE integration tests.')

from equitrain import get_args_parser_predict, predict
from equitrain.utility_test import MaceWrapper
from equitrain.utility_test.mace_support import get_mace_model_path


def test_mace_predict(tmp_path):
    args = get_args_parser_predict().parse_args([])

    data_dir = Path(__file__).with_name('data')
    args.predict_file = str(data_dir / 'valid.h5')
    args.batch_size = 5
    args.dtype = 'float32'
    args.output_dir = str(tmp_path / 'predict_mace')
    mace_model_path = get_mace_model_path()
    args.model = MaceWrapper(args, filename_model=mace_model_path)

    energy_pred, forces_pred, stress_pred = predict(args)

    output_dir = Path(args.output_dir)
    metadata = json.loads((output_dir / 'predictions.json').read_text())
    assert metadata['backend'] == 'torch'
    assert metadata['dataset'] == args.predict_file
    assert metadata['arrays_file'] == 'predictions.npz'

    with np.load(output_dir / metadata['arrays_file']) as arrays:
        assert np.allclose(arrays['energy'], energy_pred.detach().cpu().numpy())
        assert np.allclose(arrays['forces'], forces_pred.detach().cpu().numpy())
        assert np.allclose(arrays['stress'], stress_pred.detach().cpu().numpy())


if __name__ == '__main__':
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_mace_predict(Path(tmp_dir))
