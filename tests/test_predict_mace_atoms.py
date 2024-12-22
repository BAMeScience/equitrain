# %%
import pytest

import ase.io

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from equitrain import get_args_parser_predict
from equitrain import predict_atoms
from equitrain.data.atomic import AtomicNumberTable
from equitrain.utility     import set_dtype

from equitrain.utility_test import MaceWrapper

# %%

def test_predict_mace_atoms():

    set_dtype("float64")

    r = 4.5
    filename = 'data.xyz'

    args = get_args_parser_predict().parse_args()

    args.model = MaceWrapper()

    atoms_list = ase.io.read(filename, index=":")
    z_table = AtomicNumberTable(list(args.model.model.atomic_numbers.numpy()))

    energy, force, stress = predict_atoms(args.model, atoms_list, z_table, r)

    print(energy)
    print()


# %%
if __name__ == "__main__":
    test_predict_mace_atoms()
