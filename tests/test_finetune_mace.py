from contextlib import contextmanager

import torch

from equitrain import get_args_parser_train, train
from equitrain.utility_test import MaceWrapper


class FinetuneMaceWrapper(MaceWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for param in self.model.parameters():
            param.requires_grad = False  # Freeze base params

        # Create trainable deltas with same shapes
        self.deltas = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.zeros_like(p, requires_grad=True))
                for p in self.model.parameters()
            ]
        )

    def parameters(self, recurse: bool = True):
        """
        Override parameters() to return only the deltas (trainable parameters).
        """
        return self.deltas

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """
        Override named_parameters() to return only the deltas (trainable parameters).
        """
        # Use the parameter names of the deltas to mimic the original parameter names.
        return [
            (prefix + name, delta)
            for name, delta in zip(self.model._modules.keys(), self.deltas)
        ]

    @contextmanager
    def apply_deltas(self):
        original = [p.detach().clone() for p in self.model.parameters()]
        for p, d in zip(self.model.parameters(), self.deltas):
            p.add_(d)
        yield
        with torch.no_grad():
            for p, o in zip(self.model.parameters(), original):
                p.data.copy_(o)

    def forward(self, *args):
        with self.apply_deltas():
            return super().forward(*args)


def get_params_and_deltas(model):
    """
    Get the parameters and deltas of the model.
    """
    params = [param.detach().cpu().clone() for param in model.model.parameters()]
    deltas = [delta.detach().cpu().clone() for delta in model.parameters()]
    return params, deltas


def test_finetune_mace():
    args = get_args_parser_train().parse_args()

    args.train_file = 'data/train.h5'
    args.valid_file = 'data/valid.h5'
    args.test_file = 'data/train.h5'
    args.output_dir = 'test_finetune_mace'
    args.model = FinetuneMaceWrapper(args)

    args.epochs = 2
    args.batch_size = 2
    args.lr = 0.001
    args.loss_type = 'mse'
    args.weight_decay = 0.0
    args.ema = False
    args.verbose = 1
    args.tqdm = True

    params_old, deltas_old = get_params_and_deltas(args.model)

    train(args)

    params_new, deltas_new = get_params_and_deltas(args.model)

    # Check if parameters and deltas have changed
    for old, new in zip(params_old, params_new):
        if torch.abs(old - new).amin() > 1e-8:
            print('Parameters have changed after training.')
            break

    for old, new in zip(deltas_old, deltas_new):
        if torch.abs(old - new).amin() > 1e-8:
            print('Deltas have changed after training.')
            break


if __name__ == '__main__':
    test_finetune_mace()
