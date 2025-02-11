import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 'yes', '1'):
        return True
    elif value.lower() in ('false', 'no', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_common_file_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--train-file', help='Training data', type=str, default=None)
    parser.add_argument('--valid-file', help='Validation data', type=str, default=None)
    parser.add_argument('--test-file', help='Test data', type=str, default=None)
    parser.add_argument(
        '--statistics-file',
        help='Statistics file in JSON format',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--output-dir', help='Output directory for h5 files', type=str, default=''
    )
    return parser


def add_common_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        '--batch-size', help='Batch size for computation', type=int, default=16
    )
    parser.add_argument(
        '--batch-max-nodes',
        help='Maximum number of nodes per batch',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--batch-max-edges',
        help='Maximum number of edges per batch',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--batch-drop',
        help='Drop graphs if node or edge limit is reached',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--dtype',
        help='Set default dtype [float16, float32, float64]',
        type=str,
        default='float64',
    )
    parser.add_argument(
        '--workers', help='Number of data loading workers', type=int, default=4
    )
    parser.add_argument(
        '--pin-memory',
        help='Pin CPU memory in DataLoader.',
        type=str2bool,
        default=True,
    )
    parser.add_argument('--seed', help='Random seed for splits', type=int, default=123)
    return parser


def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--model', help='Path to a model file', type=str, default=None)
    parser.add_argument(
        '--model-wrapper', help='Model wrapper class [mace]', type=str, default=None
    )
    parser.add_argument(
        '--load-checkpoint', help='Load full checkpoint', type=str, default=None
    )
    parser.add_argument(
        '--load-checkpoint-model',
        help='Load model checkpoint only',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--energy-weight', help='Weight for energy loss', type=float, default=1.0
    )
    parser.add_argument(
        '--forces-weight', help='Weight for forces loss', type=float, default=1.0
    )
    parser.add_argument(
        '--stress-weight', help='Weight for stress loss', type=float, default=1.0
    )

    return parser


def add_optimizer_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        '--opt', help='Optimizer (e.g., adamw)', type=str, default='adamw'
    )
    parser.add_argument('--lr', help='Learning rate', type=float, default=0.01)
    parser.add_argument(
        '--gradient-clipping',
        help='Gradient clipping before optimization',
        type=float,
        default=10.0,
    )
    parser.add_argument(
        '--ema',
        help='Use exponential moving average (EMA) during training',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--ema-decay',
        help='Decay rate for EMA',
        type=float,
        default=0.999,
    )
    parser.add_argument('--weight-decay', help='Weight decay', type=float, default=None)
    parser.add_argument(
        '--alpha', default=0.99, type=float, help='Smoothing constant (default: 0.99)'
    )
    parser.add_argument(
        '--gamma',
        default=0.8,
        type=float,
        help='Multiplicative factor of learning rate decay (default: 0.8)',
    )
    parser.add_argument(
        '--momentum', default=0.9, type=float, help='SGD momentum (default: 0.9)'
    )
    parser.add_argument(
        '--min-lr',
        default=0.0,
        type=float,
        help='A lower bound on the learning rate of all param groups or each group respectively (default: 0.0)',
    )
    parser.add_argument(
        '--step-size',
        default=5,
        type=int,
        help='Period of learning rate decay for the StepLR scheduler',
    )
    parser.add_argument(
        '--eps',
        default=1e-8,
        type=float,
        help='Term added to the denominator to improve numerical stability (default: 1e-8)',
    )
    parser.add_argument(
        '--plateau-patience',
        type=int,
        default=2,
        help='The number of allowed epochs with no improvement after which the learning rate will be reduced (default: 2)',
    )
    parser.add_argument(
        '--plateau-factor',
        type=float,
        default=0.5,
        help='Factor by which the learning rate will be reduced. new_lr = lr * factor (default: 0.5)',
    )
    parser.add_argument(
        '--plateau-threshold',
        type=float,
        default=1e-4,
        help='Threshold for measuring the new optimum, to only focus on significant changes (default: 1e-4)',
    )
    parser.add_argument(
        '--plateau-mode',
        type=str,
        default='min',
        help='One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing (default: min)',
    )
    parser.add_argument(
        '--decay-rate',
        '--dr',
        type=float,
        default=0.5,
        help='LR decay rate (default: 0.5)',
    )
    parser.add_argument(
        '--weighted-sampler',
        help='Use a weighted sampler where the probability of drawing a sample is proportional to its prediction error',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--weighted-sampler-threshold',
        type=float,
        default=None,
        help='Use the median error for samples with a prediction error exceeding the specified threshold',
    )

    return parser


def get_args_parser(script_type: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(f'Equitrain {script_type} script')

    if script_type == 'preprocess':
        add_common_file_args(parser)
        add_common_data_args(parser)
        parser.add_argument(
            '--valid-fraction',
            help='Fraction of training set for validation',
            type=float,
            default=0.1,
        )
        parser.add_argument(
            '--compute-statistics',
            help='Estimate statistics from training data',
            action='store_true',
            default=False,
        )
        parser.add_argument(
            '--atomic-numbers', help='List of atomic numbers', type=str, default=None
        )
        parser.add_argument(
            '--atomic-energies',
            help='Dictionary of isolated atom energies',
            type=str,
            default='average',
        )
        parser.add_argument(
            '--r-max', help='Cutoff radius for graphs', type=float, default=4.5
        )
        parser.add_argument(
            '--energy-key',
            help='Key of reference energies in training xyz',
            type=str,
            default='energy',
        )
        parser.add_argument(
            '--forces-key',
            help='Key of reference forces in training xyz',
            type=str,
            default='forces',
        )
        parser.add_argument(
            '--stress-key',
            help='Key of reference stress in training xyz',
            type=str,
            default='stress',
        )

    elif script_type == 'train':
        add_common_file_args(parser)
        add_common_data_args(parser)
        add_model_args(parser)
        add_optimizer_args(parser)
        parser.add_argument(
            '--resume',
            help='Resume training from best checkpoint',
            action='store_true',
            default=False,
        )
        parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
        parser.add_argument(
            '--epochs-start', help='Number of starting epoch', type=int, default=1
        )
        parser.add_argument(
            '--scheduler', help='LR scheduler type', type=str, default='plateau'
        )
        parser.add_argument(
            '--loss-type',
            help='Type of loss function [mae, smooth-l1, mse, huber (default)]',
            type=str,
            default='huber',
        )
        parser.add_argument(
            '--loss-monitor',
            help='Comma separated list of loss types to monitor [default: mae,mse]',
            type=str,
            default='mae,mse',
        )
        parser.add_argument(
            '--smooth-l1-beta',
            help='Beta parameter for the Smooth-L1 loss (default: 1.0)',
            type=float,
            default=1.0,
        )
        parser.add_argument(
            '--huber-delta',
            help='Delta parameter for the Huber loss (default: 0.01)',
            type=float,
            default=0.01,
        )
        parser.add_argument(
            '--shuffle',
            help='Shuffle the training dataset',
            type=str2bool,
            default=True,
        )
        parser.add_argument(
            '--print-freq',
            type=int,
            default=100,
            help='Print interval during one epoch',
        )
        parser.add_argument(
            '--wandb-project', help='Wandb project name', type=str, default=None
        )
        parser.add_argument(
            '--tqdm', help='Show TQDM status bar', action='store_true', default=False
        )

    elif script_type == 'predict':
        add_common_file_args(parser)
        add_common_data_args(parser)
        add_model_args(parser)
        parser.add_argument(
            '--predict-file',
            help='File with data for which predictions should be computed',
            type=str,
            default=None,
        )

    parser.add_argument(
        '-v',
        '--verbose',
        action='count',
        default=0,
        help='Increase verbosity level (e.g., -v, -vv, -vvv)',
    )

    return parser


def get_args_parser_preprocess() -> argparse.ArgumentParser:
    return get_args_parser('preprocess')


def get_args_parser_train() -> argparse.ArgumentParser:
    return get_args_parser('train')


def get_args_parser_predict() -> argparse.ArgumentParser:
    return get_args_parser('predict')


def check_args_complete(args: argparse.ArgumentParser, script_type: str):
    expected_args = set(vars(get_args_parser(script_type).parse_args([])))
    # Get the actual arguments from the Namespace
    actual_args = set(vars(args).keys())

    # Check if all expected arguments are present and there are no extras
    if expected_args != actual_args:
        missing = expected_args - actual_args
        extra = actual_args - expected_args
        if missing:
            raise ValueError(f'Missing arguments: {missing}')
        if extra:
            raise ValueError(f'Unexpected arguments: {extra}')


class ArgumentError(ValueError):
    """Custom exception raised when invalid or missing argument is present."""

    pass


class ArgsFormatter:
    def __init__(self, args):
        """
        Initialize the ArgsFormatter with parsed arguments.
        :param args: argparse.Namespace object
        """
        self.args = vars(args)  # Convert Namespace to dictionary

    def format(self):
        """
        Format the arguments into a neatly indented string.
        :return: Formatted string of arguments
        """
        max_key_length = max(
            len(key) for key in self.args.keys()
        )  # Determine alignment width
        return ''.join(
            [
                f'  {key:<{max_key_length}} : {value}\n' if key != 'model' else ''
                for key, value in self.args.items()
            ]
        )

    def __str__(self):
        """
        Return the formatted string when the object is printed.
        """
        return f'Options:\n{self.format()}'


class ArgsFilterSimple:
    def __init__(self, allowed_types=None):
        # Default to basic types if no custom types are provided
        self.allowed_types = allowed_types or (int, float, str, bool, list)

    def is_simple(self, value):
        """Check if a value is of an allowed type."""
        return isinstance(value, self.allowed_types)

    def filter(self, args):
        """Filter the list of arguments to include only allowed types."""
        return {
            key: value for key, value in vars(args).items() if self.is_simple(value)
        }
