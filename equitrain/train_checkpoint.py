import os
import re
from pathlib import Path

import torch
from accelerate import Accelerator


def _list_checkpoint_directories(base_path: Path | str, monitor_target: str):
    pattern = rf'^.*best_{monitor_target}_epochs@([0-9]+)_e@([0-9]*\.[0-9]+)$'

    regex = re.compile(pattern)

    matching_dirs = []
    matching_vals = []

    for root, dirs, _ in os.walk(base_path):  # Walk through the directory tree
        for dir_name in dirs:
            if r := regex.match(dir_name):  # Check if directory matches the pattern
                matching_dirs.append(os.path.join(root, dir_name))
                matching_vals.append(float(r[2]))

    return matching_dirs, matching_vals


def _find_best_checkpoint(base_path: Path | str, monitor_target: str):
    dirs, vals = _list_checkpoint_directories(base_path, monitor_target)

    if len(dirs) == 0:
        return None

    min_index = vals.index(min(vals))

    return dirs[min_index]


def load_checkpoint(args, logger, accelerator: Accelerator, model_ema):
    if args.load_checkpoint is None and args.resume:
        args.load_checkpoint = _find_best_checkpoint(args.output_dir, 'val')

    if args.load_checkpoint is not None:
        if args.verbose > 0:
            logger.log(1, f'Loading checkpoint {args.load_checkpoint}...')

        accelerator.load_state(args.load_checkpoint)

        ema_path = Path(args.load_checkpoint) / 'ema.bin'

        if model_ema and ema_path.exists():
            model_ema.load_state_dict(torch.load(ema_path))

        if (
            m := re.match('.*best_[a-zA-Z]+_epochs@([0-9]+)_', args.load_checkpoint)
        ) is not None:
            args.epochs_start = int(m[1]) + 1

    args.epochs_start = max(args.epochs_start, 1)


def save_checkpoint(
    args, logger, accelerator: Accelerator, epoch, valid_loss, model, model_ema
):
    output_dir = 'best_val_epochs@{}_e@{:.4f}'.format(epoch, valid_loss['total'].avg)

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
