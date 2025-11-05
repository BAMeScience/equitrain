from pathlib import Path

from equitrain import get_args_parser_train, train
from equitrain.utility_test import MaceWrapper
from equitrain.utility_test.mace_support import get_mace_model_path


def test_train_mace(tmp_path):
    args = get_args_parser_train().parse_args([])

    data_dir = Path(__file__).with_name('data')
    args.train_file = str(data_dir / 'train.h5')
    args.valid_file = str(data_dir / 'valid.h5')
    args.test_file = None
    output_dir = tmp_path / 'train_mace'
    args.output_dir = str(output_dir)
    args.dtype = 'float32'
    mace_model_path = get_mace_model_path()
    args.model = MaceWrapper(args, filename_model=mace_model_path)

    args.epochs = 1
    args.batch_size = 1
    args.lr = 5e-4
    args.train_max_steps = 1
    args.valid_max_steps = 1
    args.workers = 0
    args.pin_memory = False
    args.verbose = 0
    args.tqdm = False

    train(args)


if __name__ == '__main__':
    test_train_mace()
