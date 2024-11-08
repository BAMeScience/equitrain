import ase.io
from ase.data import atomic_numbers

def get_atomic_numbers_from_xyz(xyz_file):
    # Initialize a set to keep track of unique atomic numbers
    atomic_number_set = set()
    
    # Read the XYZ file using ASE, which reads each frame separately
    for atoms in ase.io.read(xyz_file, index=':'):
        # Loop over each atom in the current frame
        for atom in atoms:
            # Get the atomic symbol of the atom
            symbol = atom.symbol
            # Get the atomic number from the symbol
            atomic_number = atomic_numbers[symbol]
            # Add the atomic number to the set (to avoid duplicates)
            atomic_number_set.add(atomic_number)
    
    # Convert the set to a sorted list
    atomic_numbers_list = sorted(atomic_number_set)
    return atomic_numbers_list

# Example usage
xyz_file = "/home/cmadaria/equitrain/equitrain/tests/data.xyz"  # Replace with your file path
atomic_numbers_list = get_atomic_numbers_from_xyz(xyz_file)
print("Unique atomic numbers in the XYZ file:", atomic_numbers_list)