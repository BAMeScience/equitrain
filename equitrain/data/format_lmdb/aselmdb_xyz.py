import os
import ase.io
import numpy as np
from tqdm import tqdm
from fairchem.core.datasets import AseDBDataset

# Define the dataset path
dataset_path = "/home/cmadaria/equitrain/equitrain/tests/data/lmdb/val.aselmdb"
config_kwargs = {}  # Add additional configuration if necessary

# Load the ASE LMDB dataset
dataset = AseDBDataset(config=dict(src=dataset_path, **config_kwargs))

# Create output directory for XYZ files
out_dir = f"{os.path.dirname(__file__)}/omat24_xyz"
os.makedirs(out_dir, exist_ok=True)

combined = []

# Iterate through the dataset
for i in tqdm(range(len(dataset))):
    # Get the atoms object by index
    atoms = dataset.get_atoms(i)
    
    # Define output file path for each atoms object
    xyz_path = f"{out_dir}/structure_{i}.extxyz"
    
    try:
        # Extract atomic properties if needed (forces, energy, etc.)
        if 'energy' in atoms.info:
            atoms.info['energy'] = atoms.info['energy']
        if 'forces' in atoms.arrays:
            atoms.arrays['forces'] = np.array(atoms.arrays['forces'])
        if 'magmoms' in atoms.arrays:
            atoms.arrays['magmoms'] = np.array(atoms.arrays['magmoms'])
        if 'stress' in atoms.info:
            atoms.info['stress'] = np.array(atoms.info['stress'])

        # Write atoms object to XYZ file
        ase.io.write(xyz_path, atoms, append=True, format="extxyz")

        # Append the atoms object to combined for final output
        combined.append(atoms)

    except Exception as err:
        print(f"Error processing structure {i}: {err}")

# Write the combined structures to a single XYZ file
ase.io.write(f"{out_dir}/combined_omat24.xyz", combined, format="extxyz", append=True)