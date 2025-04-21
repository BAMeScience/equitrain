import warnings
from collections.abc import Iterable
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm

from equitrain.argparser import (
    ArgsFormatter,
    ArgumentError,
    check_args_complete,
    get_loss_monitor,
)
from equitrain.data.loaders import get_dataloader
from equitrain.logger import FileLogger
from equitrain.loss import LossCollection
from equitrain.loss_fn import LossFnCollection
from equitrain.loss_metrics import LossMetrics
from equitrain.model import get_model
from equitrain.utility import set_dtype, set_seeds

warnings.filterwarnings('ignore', message=r'.*TorchScript type system.*')


def evaluate_main(
    args,
    model: torch.nn.Module,
    accelerator: Accelerator,
    dataloader: Iterable,
    max_steps: int = None,
    desc: str = 'Evaluating',
):
    loss_fn = LossFnCollection(**vars(args))

    model.eval()

    loss_metrics = LossMetrics(args)

    errors = torch.zeros(len(dataloader.dataset), device=accelerator.device)

    if max_steps is None:
        total = len(dataloader)
    else:
        total = max_steps

    with tqdm(
        dataloader,
        total=total,
        disable=not args.tqdm or not accelerator.is_main_process,
        desc=desc,
    ) as pbar:
        for step, data_list in enumerate(pbar):
            # Compute a collection of loss metrics for monitoring purposes
            loss_collection = LossCollection(
                args.loss_monitor, device=accelerator.device
            )

            with accelerator.no_sync(model):
                for data in data_list:
                    y_pred = model(data)

                    loss, error = loss_fn(y_pred, data)

                    if loss.main.isfinite():
                        loss_collection += loss
                        errors[data.idx] = error

            # Gather loss across processes for computing metrics
            loss_for_metrics = loss_collection.gather_for_metrics(accelerator)

            loss_metrics.update(loss_for_metrics)

            if accelerator.is_main_process and args.tqdm:
                pbar.set_description(
                    f'{desc} (loss={loss_metrics.main["total"].avg:.04g})'
                )

            # Stop evaluating if maximum number of steps is defined and reached
            if step >= total:
                break

    # Sum local errors across all processes
    accelerator.reduce(errors, reduction='sum')

    return loss_metrics, errors


def _evaluate_with_accelerator(args, accelerator: Accelerator):
    # Only main process should output information
    logger = FileLogger(
        log_to_file=True,
        enable_logging=accelerator.is_main_process,
        output_dir=None,
        verbosity=args.verbose,
    )
    logger.log(1, ArgsFormatter(args))

    """ Network """
    model = get_model(args, logger=logger)

    """ Data Loader """
    data_loader = get_dataloader(
        args, args.test_file, model.atomic_numbers, model.r_max, accelerator
    )

    # Prepare all components with accelerate
    model = accelerator.prepare(model)

    test_loss, _ = evaluate_main(
        args,
        model=model,
        accelerator=accelerator,
        dataloader=data_loader,
    )

    if accelerator.is_main_process:
        test_loss.log(logger, 'test', epoch=None, force=True)

    return test_loss


def evaluate_(args):
    set_dtype(args.dtype)

    if args.energy_weight == 0.0:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    else:
        ddp_kwargs = DistributedDataParallelKwargs()

    # Filter out main loss type and convert to list
    args.loss_monitor = get_loss_monitor(args)

    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )

    try:
        return _evaluate_with_accelerator(args, accelerator)

    finally:
        accelerator.end_training()


def evaluate(args):
    check_args_complete(args, 'evaluate')

    if args.test_file is None:
        raise ArgumentError('--test-file is a required argument')
    if args.model is None:
        raise ArgumentError('--model is a required argument')

    if (
        args.energy_weight == 0.0
        and args.forces_weight == 0.0
        and args.stress_weight == 0.0
    ):
        raise ArgumentError('at least one non-zero loss weight is required')

    return evaluate_(args)
