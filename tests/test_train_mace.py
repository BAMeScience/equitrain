import torch

from equitrain import get_args_parser_train
from equitrain import train

# %%

class MaceWrapper(torch.nn.Module):

    def __init__(self, model):

        super().__init__()

        self.model = model


    def forward(self, *args):
        r = self.model(*args, training=self.training)

        if isinstance(r, dict):
            energy = r['energy']
            forces = r['forces']
            stress = r['stress']
        else:
            energy, forces, stress = r

        return energy, forces, stress


# %%

def main():

    model = torch.load("mace-alex-main-branch.model")
    model = MaceWrapper(model)

    r = 4.5

    args = get_args_parser_train().parse_args()

    args.train_file      = f'data/train.h5'
    args.valid_file      = f'data/valid.h5'
    args.statistics_file = f'data/statistics.json'
    args.output_dir      = 'result'
    args.model           = model

    args.epochs     = 10
    args.batch_size = 64
    args.lr         = 0.01

    train(args)

# %%
if __name__ == "__main__":
    main()
