import re
import os

from pathlib import Path

from accelerate import Accelerator

def _list_checkpoint_directories(base_path : Path | str, monitor_target : str):

    pattern = f'^.*best_{monitor_target}_epochs@([0-9]+)_e@([0-9]*\.[0-9]+)$'

    regex = re.compile(pattern)

    matching_dirs = []
    matching_vals = []

    for root, dirs, _ in os.walk(base_path):  # Walk through the directory tree
        for dir_name in dirs:
            if r := regex.match(dir_name):  # Check if directory matches the pattern
                matching_dirs.append(os.path.join(root, dir_name))
                matching_vals.append(float(r[2]))

    return matching_dirs, matching_vals


def _find_best_checkpoint(base_path : Path | str, monitor_target : str):

    dirs, vals = _list_checkpoint_directories(base_path, monitor_target)

    if len(dirs) == 0:
        return None

    min_index = vals.index(min(vals))

    return dirs[min_index]


def load_checkpoint(args, logger, accelerator: Accelerator):

    if args.load_checkpoint is None and args.resume:

        args.load_checkpoint = _find_best_checkpoint(args.output_dir, 'val')

    if args.load_checkpoint is not None:

        if args.verbose > 0:
            logger.log(1, f'Loading checkpoint {args.load_checkpoint}...')

        accelerator.load_state(args.load_checkpoint)

        if (m := re.match('.*best_[a-zA-Z]+_epochs@([0-9]+)_', args.load_checkpoint)) is not None:
            args.epochs_start = int(m[1])+1

    if args.epochs_start < 1:
        args.epochs_start = 1
