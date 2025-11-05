"""
Test script for making predictions with a M3GNet model using Equitrain.
"""

import h5py
import torch

from equitrain import get_args_parser_predict, predict
from equitrain.utility_test import M3GNetWrapper


def test_m3gnet_predict():
    """
    Test prediction using a M3GNet model.
    This test creates a M3GNet wrapper and uses it to predict properties
    on a dataset.
    """
    # Parse arguments
    args = get_args_parser_predict().parse_args()

    # Set prediction parameters
    args.predict_file = 'data/valid.h5'
    args.batch_size = 32

    # Create the M3GNet wrapper
    args.model = M3GNetWrapper(args)

    # Make predictions
    energy_pred, forces_pred, stress_pred = predict(args)

    # Print predictions
    print('Energy predictions:')
    print(energy_pred[:5])
    print('\nForces predictions (shape):')
    print(forces_pred.shape)
    print('\nStress predictions (shape):')
    print(stress_pred.shape)

    return energy_pred, forces_pred, stress_pred


if __name__ == '__main__':
    test_m3gnet_predict()
