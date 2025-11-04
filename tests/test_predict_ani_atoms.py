"""
Test script for ANI wrapper.

This script tests the ANI wrapper by predicting energies and forces for a set of atoms.
"""

from pathlib import Path

import numpy as np
import torch
from ase import Atoms

from equitrain import get_args_parser_predict, predict_atoms
from equitrain.data.atomic import AtomicNumberTable
from equitrain.utility_test import AniWrapper


def _create_test_atoms():
    positions_water = np.array(
        [
            [0.0000, 0.0000, 0.0000],
            [0.9572, 0.0000, 0.0000],
            [-0.2390, 0.9270, 0.0000],
        ]
    )
    positions_methane = np.array(
        [
            [0.0000, 0.0000, 0.0000],
            [0.6291, 0.6291, 0.6291],
            [-0.6291, -0.6291, 0.6291],
            [-0.6291, 0.6291, -0.6291],
            [0.6291, -0.6291, -0.6291],
        ]
    )

    water = Atoms('H2O', positions=positions_water, cell=np.eye(3) * 10.0, pbc=[0, 0, 0])
    methane = Atoms('CH4', positions=positions_methane, cell=np.eye(3) * 10.0, pbc=[0, 0, 0])

    return [water, methane]


def test_predict_ani_atoms():
    """Test predicting energies and forces using the ANI wrapper."""
    torch.set_default_dtype(torch.float64)

    r_cut = 5.2

    args = get_args_parser_predict().parse_args([])
    args.model_wrapper = 'ani'
    args.workers = 0
    args.model = AniWrapper(args)

    atoms_list = _create_test_atoms()
    z_table = AtomicNumberTable(list(args.model.atomic_numbers))

    energy, force, stress = predict_atoms(
        args.model,
        atoms_list,
        z_table,
        r_cut,
        num_workers=0,
        pin_memory=False,
        batch_size=2,
    )

    assert energy is not None
    assert force is not None
    assert torch.allclose(stress, torch.zeros_like(stress))
