import os
import re
from pathlib import Path

import torch
from accelerate import Accelerator
from torch_ema import ExponentialMovingAverage

from equitrain.logger import FileLogger


def _list_checkpoint_directories(base_path: Path | str, monitor_target: str):
    pattern = rf'^.*best_{monitor_target}_epochs@([0-9]+)_e@([0-9]*\.[0-9]+)$'

    regex = re.compile(pattern)

    matching_dirs = []
    matching_vals = []
    matching_epochs = []

    for root, dirs, _ in os.walk(base_path):  # Walk through the directory tree
        for dir_name in dirs:
            if r := regex.match(dir_name):  # Check if directory matches the pattern
                matching_dirs.append(os.path.join(root, dir_name))
                matching_vals.append(float(r[2]))
                matching_epochs.append(int(r[1]))

    return matching_dirs, matching_vals, matching_epochs


def _find_best_checkpoint(base_path: Path | str, monitor_target: str):
    dirs, vals, epochs = _list_checkpoint_directories(base_path, monitor_target)

    if len(dirs) == 0:
        return None, None

    min_index = vals.index(min(vals))

    return dirs[min_index], epochs[min_index]


def _find_last_checkpoint(base_path: Path | str, monitor_target: str):
    dirs, _, epochs = _list_checkpoint_directories(base_path, monitor_target)

    if len(dirs) == 0:
        return None, None

    max_index = epochs.index(max(epochs))

    return dirs[max_index], epochs[max_index]


def load_model_state(model, state_dict_path):
    device = next(model.parameters()).device

    state_dict = torch.load(state_dict_path, weights_only=True, map_location=device)

    # If saved with DDP, keys start with 'module.'
    if any(k.startswith('module.') for k in state_dict.keys()):
        # Remove 'module.' prefix
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def load_checkpoint(
    args,
    model: torch.nn.Module,
    model_ema: ExponentialMovingAverage = None,
    accelerator: Accelerator = None,
    logger: FileLogger = None,
):
    load_checkpoint = getattr(args, 'load_checkpoint', None)
    load_checkpoint_model = getattr(args, 'load_checkpoint_model', None)
    load_best_checkpoint = getattr(args, 'load_best_checkpoint', None)
    load_last_checkpoint = getattr(args, 'load_last_checkpoint', None)
    load_best_checkpoint_model = getattr(args, 'load_best_checkpoint_model', None)
    load_last_checkpoint_model = getattr(args, 'load_last_checkpoint_model', None)
    epochs_start = getattr(args, 'epochs_start', 1)

    if load_checkpoint is None and load_best_checkpoint:
        load_checkpoint, epoch = _find_best_checkpoint(args.output_dir, 'val')

        if epoch is not None:
            epochs_start = epoch + 1

    if load_checkpoint is None and load_last_checkpoint:
        load_checkpoint, epoch = _find_last_checkpoint(args.output_dir, 'val')

        if epoch is not None:
            epochs_start = epoch + 1

    if load_checkpoint_model is None and load_best_checkpoint_model:
        load_checkpoint_model, epoch = _find_best_checkpoint(args.output_dir, 'val')
        if load_checkpoint_model is not None:
            load_checkpoint_model += '/pytorch_model.bin'

        if epoch is not None:
            args.epochs_start = epoch + 1

    if load_checkpoint_model is None and load_last_checkpoint_model:
        load_last_checkpoint_model, epoch = _find_last_checkpoint(
            args.output_dir, 'val'
        )
        if load_checkpoint_model is not None:
            load_checkpoint_model += '/pytorch_model.bin'

        if epoch is not None:
            epochs_start = epoch + 1

    if load_checkpoint is not None and accelerator is not None:
        if logger is not None:
            logger.log(1, f'Loading checkpoint {args.load_checkpoint}...')

        accelerator.load_state(load_checkpoint)

        ema_path = Path(args.load_checkpoint) / 'ema.bin'

        if model_ema and ema_path.exists():
            model_ema.load_state_dict(torch.load(ema_path, weights_only=True))

    if load_checkpoint_model is not None:
        if logger is not None:
            logger.log(1, f'Loading model checkpoint {load_checkpoint_model}...')

        load_model_state(model, load_checkpoint_model)

    if hasattr(args, 'load_checkpoint'):
        args.load_checkpoint = load_checkpoint
    if hasattr(args, 'load_checkpoint_model'):
        args.load_checkpoint_model = load_checkpoint_model
    if hasattr(args, 'load_best_checkpoint'):
        args.load_best_checkpoint = load_best_checkpoint
    if hasattr(args, 'load_last_checkpoint'):
        args.load_last_checkpoint = load_last_checkpoint
    if hasattr(args, 'load_best_checkpoint_model'):
        args.load_best_checkpoint_model = load_best_checkpoint_model
    if hasattr(args, 'load_last_checkpoint_model'):
        args.load_last_checkpoint_model = load_last_checkpoint_model
    if hasattr(args, 'epochs_start'):
        args.epochs_start = epochs_start


def save_checkpoint(
    args, logger, accelerator: Accelerator, epoch, valid_loss, model, model_ema
):
    output_dir = 'best_val_epochs@{}_e@{:0.4g}'.format(epoch, valid_loss['total'].avg)

    logger.log(
        1,
        f'Epoch [{epoch:>4}] -- Validation error decreased. Saving checkpoint to `{output_dir}`...',
    )

    # Prefix with output directory after logging
    output_dir = Path(args.output_dir) / output_dir

    accelerator.save_state(output_dir, safe_serialization=False)

    if model_ema is not None:
        torch.save(model_ema.state_dict(), output_dir / 'ema.bin')

        with model_ema.average_parameters():
            torch.save(
                model.state_dict(), output_dir / 'pytorch_model_averaged_weights.bin'
            )
