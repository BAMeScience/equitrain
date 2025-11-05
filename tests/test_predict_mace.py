from pathlib import Path

from equitrain import get_args_parser_predict, predict
from equitrain.utility_test import MaceWrapper
from equitrain.utility_test.mace_support import get_mace_model_path


def test_mace_predict():
    args = get_args_parser_predict().parse_args([])

    data_dir = Path(__file__).with_name('data')
    args.predict_file = str(data_dir / 'valid.h5')
    args.batch_size = 5
    args.dtype = 'float32'
    mace_model_path = get_mace_model_path()
    args.model = MaceWrapper(args, filename_model=mace_model_path)

    energy_pred, forces_pred, stress_pred = predict(args)

    print(energy_pred)
    print(forces_pred)
    print(stress_pred)


if __name__ == '__main__':
    test_mace_predict()
