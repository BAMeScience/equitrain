"""
Test script for ANI wrapper.

This script tests the ANI wrapper by predicting energies and forces for a set of atoms.
"""

import ase.io
import torch

from equitrain import get_args_parser_predict, predict_atoms
from equitrain.data.atomic import AtomicNumberTable
from equitrain.utility import set_dtype
from equitrain.utility_test import AniWrapper


def test_predict_ani_atoms():
    """Test predicting energies and forces using the ANI wrapper."""
    # Set data type to double precision
    set_dtype('float64')

    # Set cutoff radius
    r = 5.2
    # Path to test data
    filename = 'data.xyz'

    # Get prediction arguments
    args = get_args_parser_predict().parse_args()
    # Set model wrapper to ANI
    args.model_wrapper = 'ani'
    # Create ANI wrapper
    args.model = AniWrapper(args)

    # Load test atoms
    atoms_list = ase.io.read(filename, index=':')
    # Get atomic numbers table from model
    z_table = AtomicNumberTable(list(args.model.atomic_numbers))

    # Predict energies, forces, and stress
    energy, force, stress = predict_atoms(args.model, atoms_list, z_table, r)

    # Print results
    print('Predicted energies:')
    print(energy)
    print('\nPredicted forces:')
    print(force)
    print('\nPredicted stress:')
    print(stress)

    # Verify that energies and forces are not None
    assert energy is not None
    assert force is not None
    # ANI doesn't predict stress, so it should be zeros
    assert torch.allclose(stress, torch.zeros_like(stress))

    return energy, force, stress


if __name__ == '__main__':
    test_predict_ani_atoms()
