"""
Test script for training a M3GNet model with Equitrain.
"""

from equitrain import get_args_parser_train, train
from equitrain.utility_test import M3GNetWrapper


def test_train_m3gnet():
    """
    Test training a M3GNet model.
    This test creates a M3GNet wrapper and trains it on a small dataset.
    """
    # Parse arguments
    args = get_args_parser_train().parse_args()

    # Set training parameters
    args.train_file = 'data/train.h5'
    args.valid_file = 'data/valid.h5'
    args.output_dir = 'test_train_m3gnet'
    args.epochs = 2
    args.batch_size = 32
    args.lr = 0.001
    args.verbose = 1
    args.tqdm = True

    # Set loss weights
    args.energy_weight = 1.0
    args.forces_weight = 10.0
    args.stress_weight = 0.1

    # Create the M3GNet wrapper
    args.model = M3GNetWrapper(args)

    # Train the model
    train(args)


if __name__ == '__main__':
    test_train_m3gnet()
