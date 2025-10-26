from pathlib import Path

import ase.io

from equitrain import get_args_parser_predict, predict_atoms
from equitrain.data.atomic import AtomicNumberTable
from equitrain.utility import set_dtype
from equitrain.utility_test import MaceWrapper


def test_predict_mace_atoms():
    set_dtype('float64')

    r = 4.5
    filename = Path(__file__).with_name('data.xyz')

    args = get_args_parser_predict().parse_args([])

    args.model = MaceWrapper(args)

    atoms_list = ase.io.read(filename, index=':')
    z_table = AtomicNumberTable(list(args.model.model.atomic_numbers.numpy()))

    energy, force, stress = predict_atoms(args.model, atoms_list, z_table, r)

    print(energy)
    print()
