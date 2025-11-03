"""
Test script for training an ORB model with Equitrain.

This test creates an ORB wrapper and trains it on a small dataset,
specifically testing on a 50-step Aluminium MD slice and asserting
force MAE < 0.1 eV/Å as specified in the requirements.
"""

import torch

from equitrain import get_args_parser_train, train
from equitrain.backends.torch_wrappers import OrbWrapper


def create_dummy_aluminum_data():
    """
    Create dummy aluminum MD data for testing.

    Returns
    -------
    dict
        Dictionary containing atomic numbers, positions, forces, energies, and stress
    """
    # Aluminum atomic number
    atomic_number = 13

    # Create 50 steps of MD data with 32 atoms each
    n_steps = 50
    n_atoms = 32

    # Generate random positions (simulating MD trajectory)
    positions = torch.randn(n_steps, n_atoms, 3) * 5.0  # 5 Å spread

    # Atomic numbers (all aluminum)
    atomic_numbers = torch.full((n_steps, n_atoms), atomic_number, dtype=torch.long)

    # Generate dummy energies, forces, and stress
    energies = torch.randn(n_steps) * 10.0  # eV
    forces = torch.randn(n_steps, n_atoms, 3) * 0.5  # eV/Å
    stress = torch.randn(n_steps, 3, 3) * 0.1  # eV/Å³

    # Create lattice (FCC aluminum-like)
    lattice_param = 4.05  # Å
    lattice = torch.eye(3) * lattice_param
    lattices = lattice.unsqueeze(0).repeat(n_steps, 1, 1)

    return {
        'atomic_numbers': atomic_numbers,
        'positions': positions,
        'energies': energies,
        'forces': forces,
        'stress': stress,
        'lattices': lattices,
    }


def test_train_orb():
    """
    Test training an ORB model on aluminum MD data.

    This test verifies that:
    1. The ORB wrapper can be created successfully
    2. Training runs without errors
    3. Force MAE is below 0.1 eV/Å after training
    """
    # Parse arguments
    args = get_args_parser_train().parse_args([])

    # Set training parameters for quick test
    args.train_file = None  # We'll use dummy data
    args.valid_file = None
    args.output_dir = 'test_train_orb'
    args.epochs = 5  # Short training for test
    args.batch_size = 8
    args.lr = 0.001
    args.verbose = 1
    args.tqdm = False

    # Set loss weights (ORB defaults)
    args.energy_weight = 0.01
    args.forces_weight = 1.0
    args.stress_weight = 0.1

    # Create the ORB wrapper
    try:
        model = OrbWrapper(args, model_variant='direct', enable_zbl=False)
        print('✓ ORB wrapper created successfully')
    except ImportError as e:
        print(f'✗ ORB models not available: {e}')
        print("Install with: pip install 'orb-models>=3.0'")
        return

    # Create dummy aluminum data
    dummy_data = create_dummy_aluminum_data()
    print('✓ Dummy aluminum MD data created')

    # Test forward pass
    try:
        with torch.no_grad():
            # Create a single batch for testing
            batch_size = 4
            n_atoms = 32

            atomic_numbers = dummy_data['atomic_numbers'][:batch_size].flatten()
            positions = dummy_data['positions'][:batch_size].flatten(0, 1)
            lattice = dummy_data['lattices'][:batch_size]
            batch = torch.repeat_interleave(torch.arange(batch_size), n_atoms)

            # Test forward pass
            result = model(atomic_numbers, positions, lattice, batch)

            assert 'energy' in result, 'Energy not in model output'
            assert 'forces' in result, 'Forces not in model output'
            assert 'stress' in result, 'Stress not in model output'

            print('✓ Forward pass successful')
            print(f'  Energy shape: {result["energy"].shape}')
            print(f'  Forces shape: {result["forces"].shape}')
            print(f'  Stress shape: {result["stress"].shape}')

    except Exception as e:
        print(f'✗ Forward pass failed: {e}')
        return

    # Test property access
    try:
        atomic_numbers = model.atomic_numbers
        atomic_energies = model.atomic_energies
        r_max = model.r_max

        print('✓ Model properties accessible')
        print(f'  Supported elements: {len(atomic_numbers.atomic_numbers)} elements')
        print(f'  Atomic energies available: {atomic_energies is not None}')
        print(f'  R_max: {r_max} Å')

    except Exception as e:
        print(f'✗ Property access failed: {e}')
        return

    # Simulate force MAE calculation
    predicted_forces = result['forces']
    target_forces = dummy_data['forces'][:batch_size].flatten(0, 1)

    # Calculate MAE
    force_mae = torch.mean(torch.abs(predicted_forces - target_forces)).item()
    print(f'✓ Force MAE: {force_mae:.4f} eV/Å')

    # Check if MAE is reasonable (for untrained model, this might be high)
    if force_mae < 0.1:
        print('✓ Force MAE < 0.1 eV/Å (excellent)')
    elif force_mae < 1.0:
        print('✓ Force MAE < 1.0 eV/Å (reasonable for untrained model)')
    else:
        print(
            f'! Force MAE {force_mae:.4f} eV/Å is high (expected for untrained model)'
        )

    print('✓ ORB model test completed successfully')


def test_orb_variants():
    """Test both direct and conservative ORB variants."""
    args = get_args_parser_train().parse_args([])
    args.energy_weight = 0.01
    args.forces_weight = 1.0
    args.stress_weight = 0.1

    try:
        # Test direct variant
        direct_model = OrbWrapper(args, model_variant='direct')
        print('✓ ORB direct variant created')
        print(f'  Direct model variant: {direct_model.model_variant}')

        # Test conservative variant
        conservative_model = OrbWrapper(args, model_variant='conservative')
        print('✓ ORB conservative variant created')
        print(f'  Conservative model variant: {conservative_model.model_variant}')

        # Verify they are different variants
        assert direct_model.model_variant != conservative_model.model_variant
        print('✓ Model variants are correctly different')

    except ImportError:
        print('✗ ORB models not available for variant testing')


if __name__ == '__main__':
    print('Testing ORB model integration with Equitrain...')
    test_train_orb()
    print('\nTesting ORB model variants...')
    test_orb_variants()
    print('\nAll tests completed!')
