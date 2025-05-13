import copy

import torch

from equitrain import get_args_parser_train, train
from equitrain.utility_test import MaceWrapper


class FineTuneModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        # This is the initial readout module, where parameters are fixed
        self.finetune_fixed = module
        # The second module can be trained, ideally with a weight decay so that
        # solutions stay close to the initial module
        self.finetune_train = copy.deepcopy(module)
        # Initialize all weights to zero
        for param in self.finetune_train.parameters():
            param.data.zero_()

    def forward(self, node_feats, node_heads):
        y1 = self.finetune_fixed(node_feats, node_heads)
        y2 = self.finetune_train(node_feats, node_heads)
        return y1 + y2


class FinetuneMaceWrapper(MaceWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace readouts with fine-tuning versions
        for i, readout in enumerate(self.model.readouts):
            self.model.readouts[i] = FineTuneModule(readout)


def test_finetune_mace():
    args = get_args_parser_train().parse_args()

    args.train_file = 'data/train.h5'
    args.valid_file = 'data/valid.h5'
    args.test_file = 'data/train.h5'
    args.output_dir = 'test_finetune_mace_readout'
    args.model = FinetuneMaceWrapper(args)
    # Freeze all weights except for fine-tuning layers
    args.unfreeze_params = ['.*finetune_train.*']

    args.epochs = 10
    args.batch_size = 2
    args.lr = 0.001
    args.weight_decay = 0.001
    args.verbose = 1
    args.tqdm = True

    train(args)


if __name__ == '__main__':
    test_finetune_mace()
