import copy
from pathlib import Path

import torch

from equitrain import get_args_parser_train, train
from equitrain.utility_test import MaceWrapper
from equitrain.utility_test.mace_support import get_mace_model_path


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


def test_finetune_mace(tmp_path):
    args = get_args_parser_train().parse_args([])

    data_dir = Path(__file__).with_name('data')
    args.train_file = str(data_dir / 'train.h5')
    args.valid_file = str(data_dir / 'valid.h5')
    args.test_file = None
    output_dir = tmp_path / 'finetune_mace_readout'
    args.output_dir = str(output_dir)
    mace_model_path = get_mace_model_path()
    args.model = FinetuneMaceWrapper(args, filename_model=mace_model_path)
    # Freeze all weights except for fine-tuning layers
    args.unfreeze_params = ['.*finetune_train.*']

    args.epochs = 1
    args.batch_size = 1
    args.lr = 5e-4
    args.weight_decay = 0.0
    args.train_max_steps = 1
    args.valid_max_steps = 1
    args.workers = 0
    args.pin_memory = False
    args.verbose = 0
    args.tqdm = False

    train(args)


if __name__ == '__main__':
    test_finetune_mace()
