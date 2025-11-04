"""
Test script for ANI energy-only training.

This script tests the ANI wrapper's ability to train using only energy labels.
"""

from pathlib import Path

import numpy as np
import torch
import torchani
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from equitrain import get_args_parser_train, train
from equitrain.data.format_hdf5.dataset import HDF5Dataset


def _write_energy_only_dataset(path: Path) -> None:
    dataset = HDF5Dataset(path, mode='w')
    base_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.75, 0.58, 0.0],
            [-0.75, 0.58, 0.0],
        ],
        dtype=float,
    )

    for idx, energy in enumerate([-76.0, -75.5]):
        positions = base_positions + 0.01 * idx
        atoms = Atoms(
            symbols='H2O',
            positions=positions,
            cell=np.eye(3) * 10.0,
            pbc=[0, 0, 0],
        )
        calc = SinglePointCalculator(
            atoms,
            energy=energy,
            forces=np.zeros((len(atoms), 3), dtype=float),
            stress=np.zeros((3, 3), dtype=float),
        )
        atoms.calc = calc
        atoms.info['virials'] = np.zeros((3, 3), dtype=float)
        atoms.info['dipole'] = np.zeros(3, dtype=float)
        atoms.info['energy_weight'] = 1.0
        atoms.info['forces_weight'] = 0.0
        atoms.info['stress_weight'] = 0.0
        atoms.info['virials_weight'] = 0.0
        atoms.info['dipole_weight'] = 0.0
        dataset[idx] = atoms

    dataset.close()


def test_train_ani_energy_only(tmp_path):
    """Test training ANI model with energy-only data."""
    parser = get_args_parser_train()
    args = parser.parse_args([])

    args.model_wrapper = 'ani'
    args.energy_weight = 1.0
    args.forces_weight = 0.0
    args.stress_weight = 0.0

    args.batch_size = 32
    args.epochs = 2
    args.lr = 1e-4
    args.dtype = 'float64'
    output_dir = tmp_path / 'ani_training'
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    args.test_file = None

    train_file = output_dir / 'train.h5'
    valid_file = output_dir / 'valid.h5'
    _write_energy_only_dataset(train_file)
    _write_energy_only_dataset(valid_file)
    args.train_file = str(train_file)
    args.valid_file = str(valid_file)

    ani_model = torchani.models.ANI1x(periodic_table_index=False).double()
    ani_model.species_order = ['H', 'C', 'N', 'O']
    args.model = ani_model

    train(args)

    checkpoint_dirs = sorted(output_dir.glob('best_val_epochs@*_e@*'))
    assert checkpoint_dirs, 'No checkpoints were written.'

    latest_checkpoint = checkpoint_dirs[-1]
    assert (latest_checkpoint / 'model.safetensors').exists()
    assert (latest_checkpoint / 'optimizer.bin').exists()
    assert (latest_checkpoint / 'scheduler.bin').exists()
    assert (latest_checkpoint / 'args.json').exists()
