import numpy as np
import h5py
from ase import Atoms
from fairchem.core.datasets import AseDBDataset


def convert_to_ase_object(data):
    # Use 'pos' for atomic positions
    positions = data['pos']

    # Extract atomic numbers
    numbers = data['atomic_numbers']

    # Extract the cell and PBC (periodic boundary conditions) information
    cell = data['cell']

    # Properly handle different cell formats
    if len(cell) == 6:  # For triclinic or hexagonal systems
        atoms = Atoms(numbers=numbers, positions=positions, cell=cell[:3], pbc=data['pbc'])
        atoms.set_cell(cell)  # Setting 6-component cell (lengths + angles)
    elif len(cell) == 3:  # For orthorhombic systems
        atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=data['pbc'])
    else:
        atoms = Atoms(numbers=numbers, positions=positions, cell=np.reshape(cell, (3, 3)), pbc=data['pbc'])

    # Include energy if available
    if 'energy' in data:
        atoms.info['energy'] = data['energy']

    # Include forces if available
    if 'forces' in data:
        atoms.arrays['forces'] = data['forces']

    # Include stress if available
    if 'stress' in data:
        atoms.info['stress'] = data['stress']

    return atoms

metadata = np.load('equitrain/equitrain/tests/data/lmdb/val_metadata.npz')

dataset_path = '/home/cmadaria/equitrain/equitrain/tests/data/lmdb/val.aselmdb'
# Update config_kwargs with metadata
config_kwargs = {
    'metadata': {
        'title': 'LMDB Dataset',
        'key_descriptions': {'natoms': ('Number of atoms', 'Number of atoms in the system', 'count')},
        'natoms': metadata['natoms'].tolist()  # Include natoms in metadata
    },
    'a2g_args': {
        'r_energy': True,
        'r_forces': True,
        'r_stress': True
    }
}

dataset = AseDBDataset(config=dict(src=dataset_path, **config_kwargs))

# Initialize the HDF5 file
output_hdf5 = 'output_data.h5'

with h5py.File(output_hdf5, 'w') as f:
    # Create batches, each with multiple configurations
    batch_size = 5  # Number of configurations per batch (adjust as needed)
    current_batch = None

    for i, data in enumerate(dataset):
        # Start a new batch every 'batch_size' configurations
        if i % batch_size == 0:
            batch_index = i // batch_size
            current_batch = f.create_group(f"config_batch_{batch_index}")
        
        # Convert LMDB data to ASE Atoms object
        atoms = convert_to_ase_object(data)
        
        # Create a new configuration inside the current batch
        config_group = current_batch.create_group(f"config_{i % batch_size}")
        config_group.create_dataset('positions', data=atoms.get_positions())
        config_group.create_dataset('numbers', data=atoms.get_atomic_numbers())
        config_group.create_dataset('cell', data=atoms.get_cell())
        config_group.create_dataset('energy', data=atoms.info['energy'])
        config_group.create_dataset('forces', data=atoms.arrays['forces'])
        config_group.create_dataset('stress', data=atoms.info['stress'])

print(f"Data successfully written to {output_hdf5}")
