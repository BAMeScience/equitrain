from pathlib import Path

import ase.io
from pymatgen.io.ase import AseAtomsAdaptor

from equitrain import get_args_parser_predict, predict_structures
from equitrain.backends.torch_utils import set_dtype
from equitrain.data.atomic import AtomicNumberTable
from equitrain.utility_test import MaceWrapper


def test_predict_mace_structures(mace_model_path):
    set_dtype('float64')

    r = 4.5
    filename = Path(__file__).with_name('data.xyz')

    args = get_args_parser_predict().parse_args([])

    args.model = MaceWrapper(args, filename_model=mace_model_path)

    atoms_list = ase.io.read(filename, index=':')
    z_table = AtomicNumberTable(list(args.model.model.atomic_numbers.numpy()))

    structures_list = [AseAtomsAdaptor.get_structure(atom) for atom in atoms_list]

    energy, force, stress = predict_structures(args.model, structures_list, z_table, r)

    print(energy)
    print()
