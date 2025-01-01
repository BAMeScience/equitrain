# %%
import pytest

import ase.io

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from equitrain import get_args_parser_predict
from equitrain import predict_structures
from equitrain.data.atomic import AtomicNumberTable
from equitrain.utility     import set_dtype

from equitrain.utility_test import MaceWrapper

# %%

def test_predict_mace_structures():

    set_dtype("float64")

    r = 4.5
    filename = 'data.xyz'

    args = get_args_parser_predict().parse_args()

    args.model = MaceWrapper(args)

    atoms_list = ase.io.read(filename, index=":")
    z_table = AtomicNumberTable(list(args.model.model.atomic_numbers.numpy()))

    structures_list = [ AseAtomsAdaptor.get_structure(atom) for atom in atoms_list ]

    energy, force, stress = predict_structures(args.model, structures_list, z_table, r)

    print(energy)
    print()


# %%
if __name__ == "__main__":
    test_predict_mace_structures()
