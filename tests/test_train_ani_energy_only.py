"""
Test script for ANI energy-only training.

This script tests the ANI wrapper's ability to train using only energy labels.
"""

import os

import torch

from equitrain import get_args_parser_train, train
from equitrain.utility_test import AniWrapper


def test_train_ani_energy_only():
    """Test training ANI model with energy-only data."""
    # Create arguments for training
    parser = get_args_parser_train()
    args = parser.parse_args([])

    # Set model wrapper to ANI
    args.model_wrapper = 'ani'
    # Use energy-only training (no forces or stress)
    args.energy_weight = 1.0
    args.forces_weight = 0.0
    args.stress_weight = 0.0

    # Set paths for training data
    args.train_file = 'data/train.h5'
    args.valid_file = 'data/valid.h5'

    # Set training parameters
    args.batch_size = 32
    args.epochs = 2  # Just a few epochs for testing
    args.lr = 1e-4

    # Create ANI wrapper
    args.model = AniWrapper(args)

    # Train the model
    train(args)

    # Check that the model was trained and saved
    assert os.path.exists('checkpoint.pt')

    # Load the trained model
    checkpoint = torch.load('checkpoint.pt')

    # Verify that the checkpoint contains the expected keys
    assert 'model' in checkpoint
    assert 'optimizer' in checkpoint
    assert 'scheduler' in checkpoint
    assert 'epoch' in checkpoint

    # Clean up
    os.remove('checkpoint.pt')

    return checkpoint


if __name__ == '__main__':
    test_train_ani_energy_only()
