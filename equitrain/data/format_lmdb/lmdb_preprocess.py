from equitrain.mace.data.utils import compute_average_E0s, compute_one_hot
import numpy as np
import h5py
import json
import logging
import torch
from ase import Atoms
from fairchem.core.datasets import AseDBDataset
from collections import defaultdict
from equitrain.mace.modules.blocks import AtomicEnergiesBlock
from equitrain.mace.tools import AtomicNumberTable, to_numpy, atomic_numbers_to_indices, to_one_hot
from equitrain.mace.tools.scatter import scatter_sum


# Convert LMDB data to ASE Atoms object
def convert_to_ase_object(data):
    positions = data['pos']
    numbers = data['atomic_numbers']
    cell = data['cell']

    if len(cell) == 6:
        atoms = Atoms(numbers=numbers, positions=positions, cell=cell[:3], pbc=data['pbc'])
        atoms.set_cell(cell)
    elif len(cell) == 3:
        atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=data['pbc'])
    else:
        atoms = Atoms(numbers=numbers, positions=positions, cell=np.reshape(cell, (3, 3)), pbc=data['pbc'])

    if 'energy' in data:
        atoms.info['energy'] = data['energy']
    if 'forces' in data:
        atoms.arrays['forces'] = data['forces']
    if 'stress' in data:
        atoms.info['stress'] = data['stress']

    return atoms


# Extract atomic numbers, energies, and neighbor statistics from LMDB dataset
def extract_statistics_from_lmdb(dataset, z_table):
    atomic_energies_dict = defaultdict(float)  # Initialize dictionary for atomic energies
    forces_list = []
    num_neighbors = []
    atom_energy_list = []

    for data in dataset:
        atoms = convert_to_ase_object(data)

        atomic_numbers = atoms.get_atomic_numbers()
        energy = atoms.info.get('energy', 0.0)
        forces = atoms.arrays.get('forces', np.zeros_like(atoms.get_positions()))

        # Accumulate atomic energies for each element in the system
        for z in atomic_numbers:
            atomic_energies_dict[z] += energy / len(atomic_numbers)  # Per-atom contribution

        # Collect forces and energies for statistics
        forces_list.append(forces)
        atom_energy_list.append(energy)

        # Neighbor calculation can be handled using positions data
        positions = atoms.get_positions()
        avg_neighbors = len(positions) - 1  # Simplified neighbor calculation
        num_neighbors.append(avg_neighbors)

    # Convert to numpy arrays for statistical calculations
    forces = np.concatenate(forces_list, axis=0)
    atom_energies = np.array(atom_energy_list)

    # Compute statistical values
    avg_num_neighbors = np.mean(num_neighbors)
    mean_energy = np.mean(atom_energies)
    std_energy = np.std(atom_energies)

    return atomic_energies_dict, avg_num_neighbors, mean_energy, std_energy


# Compute statistics (average number of neighbors, mean energy, and forces)
def compute_lmdb_statistics(data_loader, atomic_energies, z_table):
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)
    atom_energy_list = []
    forces_list = []
    num_neighbors = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(compute_one_hot(z_table, batch))
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        )
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_energy_list.append((batch.y - graph_e0s) / graph_sizes)

        forces_list.append(batch.force)

        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    atom_energies = torch.cat(atom_energy_list, dim=0)
    forces = torch.cat(forces_list, dim=0)

    mean = to_numpy(torch.mean(atom_energies)).item()
    rms = to_numpy(torch.sqrt(torch.mean(torch.square(forces)))).item()

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    ).item()

    return avg_num_neighbors, mean, rms


# Main function to generate statistics.json and write to HDF5
def generate_statistics_and_convert_hdf5(lmdb_dataset, z_table, output_hdf5, output_dir, r_max):
    atoms_list = []  # Store Atoms objects for train_collection

    # Initialize the HDF5 file for conversion
    with h5py.File(output_hdf5, 'w') as f:
        # Create batches, each with multiple configurations
        batch_size = 5  # Number of configurations per batch (adjust as needed)
        current_batch = None

        forces_list = []
        num_neighbors = []
        atom_energy_list = []

        for i, data in enumerate(lmdb_dataset):
            # Start a new batch every 'batch_size' configurations
            if i % batch_size == 0:
                batch_index = i // batch_size
                current_batch = f.create_group(f"config_batch_{batch_index}")

            # Convert LMDB data to ASE Atoms object
            atoms = convert_to_ase_object(data)
            atoms_list.append(atoms)  # Add to train_collection list

            # Collect forces and energies for statistics
            forces_list.append(atoms.arrays['forces'])
            atom_energy_list.append(atoms.info['energy'])

            # Collect positions to compute the number of neighbors
            positions = atoms.get_positions()
            avg_neighbors = len(positions) - 1
            num_neighbors.append(avg_neighbors)

            # Create a new configuration inside the current batch
            config_group = current_batch.create_group(f"config_{i % batch_size}")
            config_group.create_dataset('positions', data=atoms.get_positions())
            config_group.create_dataset('numbers', data=atoms.get_atomic_numbers())
            config_group.create_dataset('cell', data=atoms.get_cell())
            config_group.create_dataset('energy', data=atoms.info['energy'])
            config_group.create_dataset('forces', data=atoms.arrays['forces'])
            config_group.create_dataset('stress', data=atoms.info['stress'])

        # Use the accumulated atoms_list as the train_collection
        atomic_energies_dict = compute_average_E0s(atoms_list, z_table)

        # Calculate statistics based on the data collected
        avg_num_neighbors = np.mean(num_neighbors)
        mean_energy = np.mean(atom_energy_list)
        std_energy = np.std(atom_energy_list)

        # Write the statistics to statistics.json
        statistics = {
            "atomic_energies": str(atomic_energies_dict),
            "avg_num_neighbors": avg_num_neighbors,
            "mean": mean_energy,
            "std": std_energy,
            "atomic_numbers": str(z_table.zs),
            "r_max": r_max
        }

        with open(f"{output_dir}/statistics.json", "w") as json_file:
            json.dump(statistics, json_file)

        print(f"Statistics and HDF5 file successfully written to {output_hdf5} and {output_dir}/statistics.json")


# Load and process the dataset and compute statistics
def main():
    # Metadata and dataset path
    metadata = np.load('equitrain/equitrain/tests/data/lmdb/val_metadata.npz')
    dataset_path = '/home/cmadaria/equitrain/equitrain/tests/data/lmdb/val.aselmdb'

    # Configurations for dataset loading
    config_kwargs = {
        'metadata': {
            'title': 'LMDB Dataset',
            'key_descriptions': {'natoms': ('Number of atoms', 'Number of atoms in the system', 'count')},
            'natoms': metadata['natoms'].tolist()
        },
        'a2g_args': {
            'r_energy': True,
            'r_forces': True,
            'r_stress': True
        }
    }

    # Load LMDB dataset
    lmdb_dataset = AseDBDataset(config=dict(src=dataset_path, **config_kwargs))

    # Create atomic number table from the dataset
    atomic_numbers = set()
    for data in lmdb_dataset:
        atoms = convert_to_ase_object(data)
        atomic_numbers.update(atoms.get_atomic_numbers())
    z_table = AtomicNumberTable(list(atomic_numbers))

    # Define output file paths
    output_hdf5 = 'output_data.h5'
    output_dir = '.'

    # r_max parameter
    r_max = 4.5  # Replace with actual r_max from args

    # Generate statistics and convert to HDF5
    generate_statistics_and_convert_hdf5(lmdb_dataset, z_table, output_hdf5, output_dir, r_max)


if __name__ == "__main__":
    main()
