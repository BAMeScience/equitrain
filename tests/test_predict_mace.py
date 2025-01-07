from equitrain import get_args_parser_predict, predict
from equitrain.utility_test import MaceWrapper


def test_mace_predict():
    args = get_args_parser_predict().parse_args()

    args.predict_file = 'data/valid.h5'
    args.statistics_file = 'data/statistics.json'
    args.batch_size = 5
    args.model = MaceWrapper(args)

    energy_pred, forces_pred, stress_pred = predict(args)

    print(energy_pred)
    print(forces_pred)
    print(stress_pred)
