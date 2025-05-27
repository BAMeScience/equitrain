import ase.io
from pymatgen.io.ase import AseAtomsAdaptor

from equitrain import get_args_parser_predict, predict_structures
from equitrain.data.atomic import AtomicNumberTable
from equitrain.utility import set_dtype
from equitrain.utility_test import M3GNetWrapper


def test_predict_m3gnet_structures():
    """
    Test prediction using M3GNet wrapper on structure data.

    This test loads structures from a file, creates a M3GNet wrapper,
    and uses it to predict energy, forces, and stress.
    """
    set_dtype('float64')

    r = 5.0  # M3GNet default cutoff
    filename = 'data.xyz'

    args = get_args_parser_predict().parse_args()

    # Create the M3GNet wrapper
    args.model = M3GNetWrapper(args)

    # Load structures from file
    atoms_list = ase.io.read(filename, index=':')
    z_table = AtomicNumberTable(list(args.model.atomic_numbers))

    # Convert ASE atoms to pymatgen structures
    structures_list = [AseAtomsAdaptor.get_structure(atom) for atom in atoms_list]

    # Predict properties
    energy, force, stress = predict_structures(args.model, structures_list, z_table, r)

    print(energy)
    print()
    print(force)
    print()
    print(stress)


if __name__ == '__main__':
    test_predict_m3gnet_structures()
