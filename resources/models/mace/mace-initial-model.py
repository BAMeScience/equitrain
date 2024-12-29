import argparse
import ast
import json
import torch

from equitrain.data import Statistics

from mace.tools import AtomicNumberTable, build_default_arg_parser, check_args
from mace.tools.model_script_utils import configure_model
from mace.tools.multihead_tools import prepare_default_head

def get_statistics():

    statistics_file = "mace-alexandria_mptraj-statistics.json"

    print(f"Reading statistics from `{statistics_file}`")

    statistics = Statistics.load(statistics_file)

    atomic_energies = [ statistics.atomic_energies[i] for i in statistics.atomic_numbers ]

    return statistics.atomic_numbers, atomic_energies, statistics.r_max, statistics.mean, statistics.std


def get_model(args: argparse.Namespace):

    # Use 64-bit precision for model weights
    torch.set_default_dtype(torch.float64)

    args.compute_energy = True
    args.compute_dipole = False

    if args.heads is not None:
        args.heads = ast.literal_eval(args.heads)
    else:
        args.heads = prepare_default_head(args)

    heads = list(args.heads.keys())

    z_table, atomic_energies, r_max, args.mean, args.std = get_statistics()

    model, output_args = configure_model(args, None, atomic_energies, model_foundation = None, heads = heads, z_table = AtomicNumberTable(z_table))

    torch.save(model, "mace-initial.model")

def main():

    arguments = [
        "--name"              , "MACE_large_density",
        "--interaction_first" , "RealAgnosticDensityInteractionBlock",
        "--interaction"       , "RealAgnosticDensityResidualInteractionBlock",
        "--num_channels"      , "128",
        "--max_L"             , "2",
        "--max_ell"           , "3",
        "--num_interactions"  , "2",
        "--correlation"       , "3",
        "--num_radial_basis"  , "8",
        "--MLP_irreps"        , "16x0e",
        "--distance_transform", "Agnesi",
        ]

    args    = build_default_arg_parser().parse_args(arguments)
    args, _ = check_args(args)

    get_model(args)


# %%
if __name__ == "__main__":
    main()
