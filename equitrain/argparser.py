import argparse


class ArgumentError(ValueError):
    """Custom exception raised when invalid or missing argument is present."""
    pass


def get_args_parser_preprocess() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Equitrain preprocess script')
    parser.add_argument("--train_file", help="Training set xyz file", type=str, default=None, required=False)
    parser.add_argument("--valid_file", help="Validation set xyz file", type=str, default=None, required=False)
    parser.add_argument(
        "--valid_fraction",
        help="Fraction of training set used for validation",
        type=float,
        default=0.1,
        required=False,
    )
    parser.add_argument(
        "--test_file",
        help="Test set xyz file",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--compute_statistics",
        help="Estimate statistics from training data",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory for h5 files",
        type=str,
        default="",
    )
    parser.add_argument(
        "--energy_key",
        help="Key of reference energies in training xyz",
        type=str,
        default="energy",
    )
    parser.add_argument(
        "--forces_key",
        help="Key of reference forces in training xyz",
        type=str,
        default="forces",
    )
    parser.add_argument(
        "--stress_key",
        help="Key of reference stress in training xyz",
        type=str,
        default="stress",
    )
    parser.add_argument(
        "--atomic_numbers",
        help="List of atomic numbers",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--batch_size", 
        help="batch size to compute average number of neighbours", 
        type=int, 
        default=16,
    )
    parser.add_argument(
        "--E0s",
        help="Dictionary of isolated atom energies",
        type=str,
        default="average",
        required=False,
    )
    parser.add_argument(
        "--r_max",
        help="Cutoff radius for computing graphs",
        type=float,
        default=4.5,
    )
    parser.add_argument(
        "--seed",
        help="Random seed for splitting training and validation sets",
        type=int,
        default=123,
    )
    parser.add_argument(
        "--dtype",
        help="Set default dtype [float16, float32, float64]",
        type=str,
        default="float64",
    )
    return parser


def get_args_parser_train():
    parser = argparse.ArgumentParser('Equitrain training script')
    # required arguments
    parser.add_argument('--train-file', type=str, default=None)
    parser.add_argument('--valid-file', type=str, default=None)
    parser.add_argument('--test-file', type=str, default=None)
    parser.add_argument('--statistics-file', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    # training hyper-parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-edge-limit", type=int, default=0,
                        help='Skip batches with too many connections, used to prevent out of memory errors')
    parser.add_argument("--eval-batch-size", type=int, default=24)
    # model parameter
    parser.add_argument('--model', type=str, default=None,
                        help='Path to a model file')
    parser.add_argument('--model-wrapper', type=str, default=None,
                        help='Model wrapper classe [mace]')
    # checkpoints
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help="Load full checkpoint including optimizer and random state (for resuming training exactly where it stopped)")
    parser.add_argument('--load-checkpoint-model', type=str, default=None,
                        help="Load model checkpoint")
    # optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Term added to the denominator to improve numerical stability (default: 1e-8)')
    parser.add_argument('--alpha', default=0.99, type=float, metavar='ALPHA',
                        help='Smoothing constant (default: 0.99)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    # learning rate schedule parameters
    parser.add_argument('--scheduler', default='plateau', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "plateau"')
    parser.add_argument('--min-lr', type=float, default=0.0, metavar='LR',
                        help='A lower bound on the learning rate of all param groups or each group respectively (default: 0.0)')
    parser.add_argument('--gamma', type=float, default=0.0, metavar='LR',
                        help='Multiplicative factor of learning rate decay (default: 0.1)')
    parser.add_argument('--plateau-patience', type=int, default=2, metavar='N',
                        help='The number of allowed epochs with no improvement after which the learning rate will be reduced (default: 2)')
    parser.add_argument('--plateau-factor', type=float, default=0.5, metavar='F',
                        help='Factor by which the learning rate will be reduced. new_lr = lr * factor (default: 0.5)')
    parser.add_argument('--plateau-threshold', type=float, default=1e-4, metavar='T',
                        help='Threshold for measuring the new optimum, to only focus on significant changes (default: 1e-4)')
    parser.add_argument('--plateau-mode', type=str, default='min', metavar='T',
                        help='One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing (default: min)')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.5, metavar='RATE',
                        help='LR decay rate (default: 0.5)')
    # logging
    parser.add_argument("--print-freq", type=int, default=100)
    # weights
    parser.add_argument('--energy-weight', help='Weight for the energy loss term (default: 0.2)', type=float, default=0.2)
    parser.add_argument('--force-weight' , help='Weight for the forces loss term (default: 0.8)', type=float, default=0.8)
    parser.add_argument('--stress-weight', help='Weight for the stress loss term (default: 0.0)', type=float, default=0.0)
    # random
    parser.add_argument("--seed", type=int, default=1)
    # data loader config
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--shuffle", help="Shuffle the training dataset", type=bool, default=True)
    parser.add_argument(
        "--dtype",
        help="Set default dtype [float16, float32, float64]",
        type=str,
        default="float64",
    )
    return parser


def get_args_parser_test() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Equitrain test script')
    # model parameter
    parser.add_argument('--model', type=str, default=None,
                        help='Path to a model file')
    parser.add_argument('--model-wrapper', type=str, default=None,
                        help='Model wrapper classe [mace]')
    # checkpoints
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help="Load full checkpoint including optimizer and random state (for resuming training exactly where it stopped)")
    parser.add_argument('--load-checkpoint-model', type=str, default=None,
                        help="Load model checkpoint")

    parser.add_argument("--test_file", help="File on which to make predictions", type=str, default=None, required=False)
    parser.add_argument(
        "--output_dir",
        help="Output directory for h5 files",
        type=str,
        default="",
    )
    parser.add_argument(
        "--energy_key",
        help="Key of reference energies in training xyz",
        type=str,
        default="energy",
    )
    parser.add_argument(
        "--forces_key",
        help="Key of reference forces in training xyz",
        type=str,
        default="forces",
    )
    parser.add_argument(
        "--stress_key",
        help="Key of reference stress in training xyz",
        type=str,
        default="stress",
    )
    parser.add_argument(
        "--batch_size", 
        help="batch size to compute average number of neighbours", 
        type=int, 
        default=16,
    )
    parser.add_argument(
        "--dtype",
        help="Set default dtype [float16, float32, float64]",
        type=str,
        default="float64",
    )
    # data loader config
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)

    return parser


def get_args_parser_predict() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Equitrain predict script')
    # model parameter
    parser.add_argument('--model', type=str, default=None,
                        help='Path to a model file')
    parser.add_argument('--model-wrapper', type=str, default=None,
                        help='Model wrapper classe [mace]')
    # checkpoints
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help="Load full checkpoint including optimizer and random state (for resuming training exactly where it stopped)")
    parser.add_argument('--load-checkpoint-model', type=str, default=None,
                        help="Load model checkpoint")

    parser.add_argument("--predict_file", help="File on which to make predictions", type=str, default=None, required=False)
    parser.add_argument(
        "--batch_size", 
        help="batch size to compute average number of neighbours", 
        type=int, 
        default=16,
    )
    # data loader config
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument(
        "--dtype",
        help="Set default dtype [float16, float32, float64]",
        type=str,
        default="float64",
    )
    parser.set_defaults(pin_mem=True)

    return parser
