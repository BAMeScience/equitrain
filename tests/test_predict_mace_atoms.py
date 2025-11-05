from pathlib import Path

import ase.io

from equitrain import get_args_parser_predict, predict_atoms
from equitrain.backends.torch_utils import set_dtype
from equitrain.data.atomic import AtomicNumberTable
from equitrain.utility_test import MaceWrapper
from equitrain.utility_test.mace_support import get_mace_model_path


def test_predict_mace_atoms():
    set_dtype('float64')

    r = 4.5
    filename = Path(__file__).with_name('data.xyz')

    args = get_args_parser_predict().parse_args([])

    mace_model_path = get_mace_model_path()
    args.model = MaceWrapper(args, filename_model=mace_model_path)

    atoms_list = ase.io.read(filename, index=':')
    z_table = AtomicNumberTable(list(args.model.model.atomic_numbers.numpy()))

    energy, force, stress = predict_atoms(
        args.model,
        atoms_list,
        z_table,
        r,
        num_workers=0,
    )

    print(energy)
    print()
