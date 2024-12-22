import h5py
import os

# Get the current working directory
current_directory = os.getcwd()

# Load both files
output_file_path = 'equitrain/equitrain/tests/data/lmdb/output_datav2.h5'
train_file_path = 'equitrain/equitrain/tests/data/train.h5'

# Function to load and check the keys and structure of the file
def inspect_h5_file(file_path, num_keys=5):
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())[:num_keys]
        structures = {key: list(f[key].keys()) if isinstance(f[key], h5py.Group) else 'Dataset' for key in keys}
    return structures

# Inspect both files (taking first few keys and structure)
output_data_structure = inspect_h5_file(output_file_path)
train_data_structure = inspect_h5_file(train_file_path)

print(output_data_structure)
print(train_data_structure)