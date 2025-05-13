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
            [torch.nn.Parameter(torch.zeros_like(p)) for p in self.model.parameters()]
        )

    def forward(self, *args):
        # Apply base layer with perturbed weights
        with torch.no_grad():
            original_params = [p.clone() for p in self.model.parameters()]

        # Temporarily patch base layer weights with (theta_0 + delta)
        for p, delta in zip(self.model.parameters(), self.deltas):
            p.data = p.data + delta

        y = self.model(*args)

        # Restore original weights
        for p, orig in zip(self.model.parameters(), original_params):
            p.data = orig.data

        print(y)

        return y


def test_finetune_mace():
    args = get_args_parser_train().parse_args()

    args.train_file = 'data/train.h5'
    args.valid_file = 'data/valid.h5'
    args.test_file = 'data/train.h5'
    args.output_dir = 'test_finetune_mace'
    args.model = FinetuneMaceWrapper(args)

    args.epochs = 10
    args.batch_size = 2
    args.lr = 0.001
    args.weight_decay = 0.001
    args.verbose = 1
    args.tqdm = True

    train(args)


if __name__ == '__main__':
    test_finetune_mace()
