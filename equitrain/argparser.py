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
        '--r-max',
        help='Override model cutoff radius for graphs (default: None)',
        type=float,
        default=None,
    )
    return parser


def add_model_checkpoint_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument(
        '--load-best-checkpoint-model',
        help='Load only the model weights from the best checkpoint (ignores optimizer and scheduler states)',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--load-last-checkpoint-model',
        help='Load only the model weights from the last checkpoint (ignores optimizer and scheduler states)',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--load-checkpoint-model',
        help='Load only the model weights from given checkpoint directory (ignores optimizer and scheduler states)',
        type=str,
        default=None,
    )
    return parser


def add_checkpoint_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument(
        '--load-best-checkpoint',
        help='Load model, optimizer, and scheduler states from best checkpoint',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--load-last-checkpoint',
        help='Load model, optimizer, and scheduler states from last checkpoint',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--load-checkpoint',
        help='Load model, optimizer, and scheduler states from given checkpoint directory',
        type=str,
        default=None,
    )
    return parser


def add_loss_weights_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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


def add_loss_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = add_loss_weights_args(parser)
    parser.add_argument(
        '--loss-type',
        help='Type of loss function [mae, smooth-l1, mse, huber (default)]',
        choices=['mae', 'smooth-l1', 'mse', 'huber'],
        type=str,
        default='huber',
    )
    parser.add_argument(
        '--loss-type-energy',
        help='Type of loss function for energy [mae, smooth-l1, mse, huber]. If not set, defaults to --loss-type.',
        choices=['mae', 'smooth-l1', 'mse', 'huber'],
        type=str,
        default=None,
    )
    parser.add_argument(
        '--loss-type-forces',
        help='Type of loss function for forces [mae, smooth-l1, mse, huber]. If not set, defaults to --loss-type.',
        choices=['mae', 'smooth-l1', 'mse', 'huber'],
        type=str,
        default=None,
    )
    parser.add_argument(
        '--loss-type-stress',
        help='Type of loss function for stress [mae, smooth-l1, mse, huber]. If not set, defaults to --loss-type.',
        choices=['mae', 'smooth-l1', 'mse', 'huber'],
        type=str,
        default=None,
    )
    parser.add_argument(
        '--loss-weight-type',
        help='Type of loss weighting scheme to apply [groundstate]. If not set, no weighting is applied.',
        choices=['groundstate'],
        type=str,
        default=None,
    )
    parser.add_argument(
        '--loss-weight-type-energy',
        help='Type of loss weighting scheme for energy [groundstate]. If not set, defaults to --loss-weight-type.',
        choices=['groundstate'],
        type=str,
        default=None,
    )
    parser.add_argument(
        '--loss-weight-type-forces',
        help='Type of loss weighting scheme for forces [groundstate]. If not set, defaults to --loss-weight-type.',
        choices=['groundstate'],
        type=str,
        default=None,
    )
    parser.add_argument(
        '--loss-weight-type-stress',
        help='Type of loss weighting scheme for stress [groundstate]. If not set, defaults to --loss-weight-type.',
        choices=['groundstate'],
        type=str,
        default=None,
    )
    parser.add_argument(
        '--loss-monitor',
        help='Comma separated list of loss types to monitor in addition to the loss function [default: mae,mse]',
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
        '--loss-clipping',
        help='Clips the loss per sample to prevent extreme outliers from disproportionately influencing the overall loss (default: None)',
        type=float,
        default=None,
    )
    return parser


def add_model_freeze_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        '--freeze-params',
        type=str,
        nargs='*',
        help='List of regex patterns matching model parameters to freeze.',
    )
    parser.add_argument(
        '--unfreeze-params',
        type=str,
        nargs='*',
        help='List of regex patterns matching model parameters to keep trainable (all others will be frozen).',
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
        '--plateau-threshold-mode',
        choices=['rel', 'abs'],
        type=str,
        default='rel',
        help='One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in `max` mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: `rel`.',
    )
    parser.add_argument(
        '--plateau-mode',
        choices=['min', 'max'],
        type=str,
        default='min',
        help='One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing (default: min)',
    )
    parser.add_argument(
        '--plateau-eps',
        default=1e-12,
        type=float,
        help='Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored (default: 1e-12)',
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


def add_inspect_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = add_model_args(parser)
    parser = add_model_checkpoint_args(parser)
    parser = add_model_freeze_args(parser)

    parser.add_argument('--output-dir', help='Output directory', type=str, default='')

    return parser


def add_export_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = add_model_args(parser)
    parser = add_model_checkpoint_args(parser)

    parser.add_argument(
        '--model-export', help='Export model to given file', type=str, default=None
    )
    parser.add_argument('--output-dir', help='Output directory', type=str, default='')

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
            '--r-max',
            help='Cutoff radius for graphs (default: 4.5)',
            type=float,
            default=4.5,
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
        parser.add_argument(
            '--output-dir', help='Output directory', type=str, default=''
        )

    elif script_type == 'train':
        add_common_file_args(parser)
        add_common_data_args(parser)
        add_model_args(parser)
        add_model_checkpoint_args(parser)
        add_model_freeze_args(parser)
        add_loss_args(parser)
        add_checkpoint_args(parser)
        add_optimizer_args(parser)

        parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
        parser.add_argument(
            '--epochs-start', help='Number of starting epoch', type=int, default=1
        )
        parser.add_argument(
            '--train-max-steps',
            help='Maximum number of steps within each training epoch (default: None)',
            type=int,
            default=None,
        )
        parser.add_argument(
            '--valid-max-steps',
            help='Maximum number of steps for computing the validation error (default: None)',
            type=int,
            default=None,
        )
        parser.add_argument(
            '--scheduler', help='LR scheduler type', type=str, default='plateau'
        )
        parser.add_argument(
            '--scheduler_monitor',
            help='Loss monitored by the scheduler [train (default), val]',
            choices=['train', 'val'],
            type=str,
            default='train',
        )
        parser.add_argument(
            '--shuffle',
            help='Shuffle the training dataset',
            type=str2bool,
            default=True,
        )
        parser.add_argument(
            '--find-unused-parameters',
            help='Find unused parameters in the model',
            action='store_true',
            default=False,
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
        parser.add_argument(
            '--output-dir', help='Output directory', type=str, default=''
        )

    elif script_type == 'predict':
        add_common_file_args(parser)
        add_common_data_args(parser)
        add_model_args(parser)
        add_loss_weights_args(parser)
        parser.add_argument(
            '--predict-file',
            help='File with data for which predictions should be computed',
            type=str,
            default=None,
        )

    elif script_type == 'evaluate':
        add_common_data_args(parser)
        add_model_args(parser)
        add_loss_args(parser)
        parser.add_argument(
            '--test-file', help='File with test data', type=str, default=None
        )
        parser.add_argument(
            '--shuffle',
            help='Shuffle the training dataset',
            type=str2bool,
            default=False,
        )
        parser.add_argument(
            '--tqdm', help='Show TQDM status bar', action='store_true', default=False
        )

    elif script_type == 'inspect':
        add_inspect_args(parser)

    elif script_type == 'export':
        add_export_args(parser)

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


def get_args_parser_evaluate() -> argparse.ArgumentParser:
    return get_args_parser('evaluate')


def get_args_parser_predict() -> argparse.ArgumentParser:
    return get_args_parser('predict')


def get_args_parser_inspect() -> argparse.ArgumentParser:
    return get_args_parser('inspect')


def get_args_parser_export() -> argparse.ArgumentParser:
    return get_args_parser('export')


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


def get_loss_monitor(args: argparse.ArgumentParser) -> list[str]:
    # Create list of loss types
    loss_monitor = [item.strip().lower() for item in args.loss_monitor.split(',')]

    if args.loss_type in loss_monitor:
        loss_monitor.remove(args.loss_type)

    return loss_monitor


def check_args_consistency(args, args_new, logger):
    for k, v in args_new.items():
        if k == 'epochs_start':
            continue
        if k == 'verbose':
            continue
        if hasattr(args, k) and getattr(args, k) != v:
            logger.log(
                1,
                f'Warning: Argument `{k}` in saved checkpoint differs from current argument: '
                f'{getattr(args, k)} != {v}.',
            )


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
