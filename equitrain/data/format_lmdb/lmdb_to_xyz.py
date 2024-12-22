import numpy as np
from ase import Atoms
from fairchem.core.datasets import AseDBDataset


def convert_to_ase_object(data):
    """Convert the LMDB data object to ASE Atoms object."""
    positions = data['pos']
    atomic_numbers = data['atomic_numbers']
    cell = data['cell']

    # Create the ASE Atoms object
    if len(cell) == 6:  # For triclinic or hexagonal systems
        atoms = Atoms(numbers=atomic_numbers, positions=positions, cell=cell[:3], pbc=data['pbc'])
        atoms.set_cell(cell)  # Setting 6-component cell (lengths + angles)
    elif len(cell) == 3:  # For orthorhombic systems
        atoms = Atoms(numbers=atomic_numbers, positions=positions, cell=cell, pbc=data['pbc'])
    else:  # Handle any other case
        atoms = Atoms(numbers=atomic_numbers, positions=positions, cell=np.reshape(cell, (3, 3)), pbc=data['pbc'])

    # Include energy, forces, and stress if available
    if 'energy' in data:
        atoms.info['energy'] = data['energy']
    if 'forces' in data:
        atoms.arrays['forces'] = data['forces']
    if 'stress' in data:
        atoms.info['stress'] = data['stress']  # Ensure stress is set correctly

    return atoms


def convert_aselmdb_to_xyz(lmdb_path, output_xyz):
    """Convert ASELMDB data to XYZ format."""
    # Load the LMDB dataset using Fairchem
    dataset = AseDBDataset(config=dict(src=lmdb_path))

    with open(output_xyz, 'w') as xyz_file:
        for i, data in enumerate(dataset):
            # Convert LMDB data to ASE Atoms object
            atoms = convert_to_ase_object(data)

            # Number of atoms
            num_atoms = len(atoms)
            xyz_file.write(f"{num_atoms}\n")

            # Lattice parameters
            cell = atoms.get_cell().reshape(-1)
            lattice_str = " ".join(map(str, cell))

            # Energy and stress
            energy = atoms.info.get('energy', 0.0)
            stress = atoms.info.get('stress', np.zeros(9))

            # Ensure stress is a 9-long vector
            if stress.shape != (9,):
                stress = np.zeros(9)  # Default to zero if stress is not valid

            # Write the header line in the XYZ file
            xyz_file.write(f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3:forces:R:3 ')
            xyz_file.write(f'config_type=Default energy={energy:.6f} energy_corrected={energy:.6f} ')
            xyz_file.write(f'stress="{ " ".join(map(str, stress)) }" pbc="T T T"\n')

            # Write each atom's data (species, position, forces)
            for symbol, pos, force in zip(atoms.get_chemical_symbols(), atoms.get_positions(), atoms.arrays.get('forces', np.zeros_like(atoms.get_positions()))):
                pos_str = " ".join(map(str, pos))
                force_str = " ".join(map(str, force))
                xyz_file.write(f"{symbol} {pos_str} {force_str}\n")

    print(f"Conversion complete: {output_xyz}")


# Example usage:
if __name__ == "__main__":
    convert_aselmdb_to_xyz('/home/cmadaria/equitrain/equitrain/tests/data/lmdb/val.aselmdb',
                            '/home/cmadaria/equitrain/equitrain/tests/data/lmdb/output_data.xyz')
