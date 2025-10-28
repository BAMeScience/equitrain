from contextlib import contextmanager
from pathlib import Path

import torch

from equitrain import get_args_parser_train, train
from equitrain.checkpoint import load_checkpoint
from equitrain.utility_test import MaceWrapper


class FinetuneMaceWrapper(MaceWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for param in self.model.parameters():
            param.requires_grad = False  # Freeze base params

        # Create trainable deltas with same shapes
        self.deltas = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros_like(p, requires_grad=True))
            for p in self.model.parameters()
        ])

    def parameters(self, recurse: bool = True):
        """
        Override parameters() to return only the deltas (trainable parameters).
        """
        return iter(self.deltas)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """
        Override named_parameters() to return only the deltas (trainable parameters).
        """
        # Use the parameter names of the deltas to mimic the original parameter names.
        return iter([
            (prefix + name, delta)
            for name, delta in zip(self.model._modules.keys(), self.deltas)
        ])

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

    def export(self, filename):
        with self.apply_deltas():
            torch.save(self.model, filename)


def get_params_and_deltas(model):
    """
    Get the parameters and deltas of the model.
    """
    params = [param.detach().cpu().clone() for param in model.model.parameters()]
    deltas = [delta.detach().cpu().clone() for delta in model.parameters()]
    return params, deltas


def save_result(args, filename):
    # Import weights from the best checkpoint
    args.load_best_checkpoint_model = True

    load_checkpoint(
        args,
        args.model,
    )
    # Add delta weights to the original model and export
    # to a new file
    args.model.export(filename)


def test_finetune_mace(tmp_path, mace_model_path):
    args = get_args_parser_train().parse_args([])

    data_dir = Path(__file__).with_name('data')
    args.train_file = str(data_dir / 'train.h5')
    args.valid_file = str(data_dir / 'valid.h5')
    args.test_file = None
    output_dir = tmp_path / 'finetune_mace'
    args.output_dir = str(output_dir)
    args.model = FinetuneMaceWrapper(args, filename_model=mace_model_path)

    args.epochs = 1
    args.batch_size = 1
    args.lr = 5e-4
    args.train_max_steps = 1
    args.valid_max_steps = 1
    args.workers = 0
    args.pin_memory = False
    args.verbose = 0
    args.tqdm = False
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

    save_result(args, str(tmp_path / 'finetune_mace.model'))


if __name__ == '__main__':
    test_finetune_mace()
