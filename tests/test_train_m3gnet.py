"""
Test script for training a M3GNet model with Equitrain.
"""

import shutil
from pathlib import Path

import pytest

pytest.importorskip('matgl', reason='matgl package is required for M3GNet tests')
pytest.importorskip('dgl', reason='dgl package is required for M3GNet tests')

from equitrain import get_args_parser_train, train
from equitrain.utility_test import M3GNetWrapper


def test_train_m3gnet():
    """
    Test training a M3GNet model.
    This test creates a M3GNet wrapper and trains it on a small dataset.
    """
    # Parse arguments
    args = get_args_parser_train().parse_args()
    args.dtype = 'float32'

    # Set training parameters
    args.train_file = 'data/train.h5'
    args.valid_file = 'data/valid.h5'
    output_dir = Path(__file__).with_name('test_train_m3gnet')
    if output_dir.exists():
        shutil.rmtree(output_dir)
    args.output_dir = str(output_dir)
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
    try:
        train(args)
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)


if __name__ == '__main__':
    test_train_m3gnet()
