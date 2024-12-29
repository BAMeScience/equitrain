import torch

from pathlib import Path

from equitrain import get_args_parser_train
from equitrain import train

# %%

def main():

    data_dir = Path("alexandria_mptraj")

    args = get_args_parser_train().parse_args()

    args.train_file      = data_dir / 'train.h5'
    args.valid_file      = data_dir / 'valid.h5'
    args.statistics_file = data_dir / 'statistics.json'
    args.output_dir      = 'result'
    args.model           = 'mace-initial.model'
    args.model_wrapper   = 'mace'
    args.wandb_project   = 'mace-alex-mptraj'

    args.energy_weight = 1.0
    args.forces_weight = 10.0
    args.stress_weight = 0.0
    
    args.epochs     = 100
    args.batch_size = 16
    args.lr         = 0.002
    args.verbose    = 1
    args.tqdm       = True

    train(args)

# %%
if __name__ == "__main__":
    main()
