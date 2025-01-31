import time
import warnings
from collections.abc import Iterable
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from equitrain.argparser import (
    ArgsFilterSimple,
    ArgsFormatter,
    ArgumentError,
    check_args_complete,
)
from equitrain.data.loaders import dataloader_update_errors, get_dataloaders
from equitrain.logger import FileLogger
from equitrain.loss import GenericLossFn, Loss
from equitrain.model import get_model
from equitrain.train_checkpoint import load_checkpoint, save_checkpoint
from equitrain.train_metrics import BestMetric, LossMetric
from equitrain.train_optimizer import create_optimizer, update_weight_decay
from equitrain.train_scheduler import SchedulerWrapper, create_scheduler
from equitrain.utility import set_dtype, set_seeds

warnings.filterwarnings('ignore', message=r'.*TorchScript type system.*')


def evaluate_main(
    args,
    model: torch.nn.Module,
    accelerator: Accelerator,
    criterion: torch.nn.Module,
    dataloader: Iterable,
    logger: FileLogger,
    desc: str = 'Evaluating',
):
    model.eval()
    criterion.eval()

    loss_metrics = LossMetric(args)

    errors = torch.zeros(len(dataloader.dataset), device=accelerator.device)

    with tqdm(
        dataloader,
        total=len(dataloader),
        disable=not args.tqdm or not accelerator.is_main_process,
        desc=desc,
    ) as pbar:
        for data_list in pbar:
            loss_sum = Loss(device=accelerator.device)

            with accelerator.no_sync(model):
                for data in data_list:
                    y_pred = model(data)

                    loss, error = criterion(y_pred, data)

                    if loss.isnan():
                        logger.log(1, 'Nan value detected. Skipping batch...')
                        continue

                    loss_sum += loss.detach()

                    errors[data.idx] = error

            # Gather loss across processes for computing metrics
            loss_for_metrics, skip = loss_sum.gather_for_metrics(accelerator)

            # Check if loss was NaN for all iterations
            if skip['total']:
                continue

            loss_metrics.update(loss_for_metrics)

            if accelerator.is_main_process and args.tqdm:
                pbar.set_description(
                    f'{desc} (loss={loss_metrics.metrics["total"].avg:.04f})'
                )

    # Synchronize updates across processes
    accelerator.wait_for_everyone()
    # Sum local errors across all processes
    accelerator.reduce(errors, reduction='sum')

    return loss_metrics, errors


def evaluate(
    args,
    model: torch.nn.Module,
    model_ema: ExponentialMovingAverage,
    *args_other,
    **kwargs,
):
    if model_ema:
        with model_ema.average_parameters():
            return evaluate_main(args, model, *args_other, **kwargs)
    else:
        return evaluate_main(args, model, *args_other, **kwargs)


def train_one_epoch(
    args,
    model: torch.nn.Module,
    model_ema: ExponentialMovingAverage,
    accelerator: Accelerator,
    criterion: torch.nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    errors: torch.Tensor,
    epoch: int,
    print_freq: int,
    logger: FileLogger,
):
    model.train()
    criterion.train()

    loss_metrics = LossMetric(args)

    start_time = time.perf_counter()

    if errors is None:
        errors = torch.zeros(len(dataloader.dataset), device=accelerator.device)
    else:
        dataloader = dataloader_update_errors(
            args, dataloader, errors, accelerator, logger
        )

    with tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        disable=not args.tqdm or not accelerator.is_main_process,
        desc='Training',
    ) as pbar:
        for step, data_list in pbar:
            loss_sum = Loss(device=accelerator.device)

            # Sub-batching causes deadlocks when the number of sub-batches varies between
            # processes. We need to loop over sub-batches withouth sync
            with accelerator.no_sync(model):
                for i_, data in enumerate(data_list):
                    try:
                        y_pred = model(data)

                        loss, error = criterion(y_pred, data)

                        if loss.isnan():
                            logger.log(2, 'Nan value detected. Skipping batch...')
                            continue

                        # Backpropagate here to prevent out-of-memory errors, gradients
                        # will be accumulated. Since we accumulate gradients over sub-batches,
                        # we have to rescale before the backward pass
                        accelerator.backward(loss['total'].value / len(data_list))

                        loss_sum += loss.detach()

                        errors[data.idx] = error

                    except torch.OutOfMemoryError:
                        logger.log(
                            1, f'OOM error on graph {i_} in batch {step}. Skipping...'
                        )
                        torch.cuda.empty_cache()

            # Gather loss across processes for computing metrics
            loss_for_metrics, skip = loss_sum.gather_for_metrics(accelerator)

            # Check if loss was NaN for all iterations in one of the processes
            if skip['total']:
                optimizer.zero_grad()
                continue

            # Clip gradients before optimization step
            if args.gradient_clipping is not None and args.gradient_clipping > 0:
                accelerator.clip_grad_value_(model.parameters(), args.gradient_clipping)

            # Sync of gradients across processes occurs here
            optimizer.step()
            optimizer.zero_grad()

            if model_ema is not None:
                model_ema.update()

            loss_metrics.update(loss_for_metrics)

            if accelerator.is_main_process:
                # Print intermediate performance statistics only for higher verbose levels
                if args.verbose > 1 and (
                    step % print_freq == 0 or step == len(dataloader) - 1
                ):
                    w = time.perf_counter() - start_time
                    e = (step + 1) / len(dataloader)

                    loss_metrics.log_step(
                        logger,
                        epoch,
                        step,
                        len(dataloader),
                        time=(1e3 * w / e / len(dataloader)),
                        lr=optimizer.param_groups[0]['lr'],
                    )

                if args.tqdm:
                    pbar.set_description(
                        f'Training (lr={optimizer.param_groups[0]["lr"]:.0e}, loss={loss_metrics.metrics["total"].avg:.04f})'
                    )

    # Synchronize updates across processes
    accelerator.wait_for_everyone()
    # Sum local errors across all processes
    accelerator.reduce(errors, reduction='sum')

    return loss_metrics, errors


def _train_with_accelerator(args, accelerator: Accelerator):
    # Only main process should output information
    logger = FileLogger(
        log_to_file=True,
        enable_logging=accelerator.is_main_process,
        output_dir=args.output_dir,
        verbosity=args.verbose,
    )
    logger.log(1, ArgsFormatter(args))

    """ Data Loader """
    train_loader, val_loader, test_loader = get_dataloaders(
        args, accelerator, logger=logger
    )

    """ Network """
    model = get_model(args, logger=logger)

    if accelerator.is_main_process:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.log(1, f'Number of params: {n_parameters}')

    """ Optimizer and LR Scheduler """
    optimizer = create_optimizer(args, model)
    lr_scheduler = create_scheduler(args, optimizer)

    criterion = GenericLossFn(**vars(args))

    # Prepare all components with accelerate
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # Exponential moving average (EMA)
    if args.ema:
        model_ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
    else:
        model_ema = None

    # Use a wrapper for the scheduler with a unified interface
    lr_scheduler = SchedulerWrapper(args, lr_scheduler)

    # Import model, optimizer, lr_scheduler from checkpoint if possible
    load_checkpoint(args, logger, accelerator, model_ema)

    # Allow users to modify the weight decay even when resuming from a checkpoint
    update_weight_decay(args, logger, optimizer)

    # Record the best validation loss and corresponding epoch
    best_metrics = BestMetric(args)

    # Prediction errors for sampling training data accordingly
    if args.weighted_sampler:
        _, errors = evaluate(
            args,
            model=model,
            model_ema=model_ema,
            accelerator=accelerator,
            criterion=criterion,
            dataloader=train_loader,
            logger=logger,
            desc='Estimating errors',
        )
    else:
        errors = None

    if args.wandb_project is not None:
        accelerator.init_trackers(
            args.wandb_project, config=ArgsFilterSimple().filter(args)
        )

    # Evaluate model before training
    if True:
        val_loss, _ = evaluate(
            args,
            model=model,
            model_ema=model_ema,
            accelerator=accelerator,
            criterion=criterion,
            dataloader=val_loader,
            logger=logger,
        )

        best_metrics.update(val_loss, args.epochs_start - 1)

        accelerator.log(
            {'val_loss': val_loss.metrics['total'].avg}, step=args.epochs_start - 1
        )

        if accelerator.is_main_process:
            val_loss.log(logger, 'val', epoch=args.epochs_start - 1)

        # Scheduler step before the first epoch for schedulers depending on the epoch
        if lr_scheduler is not None:
            lr_scheduler.step(metric=None, epoch=args.epochs_start - 1)

            last_lr = lr_scheduler.get_last_lr()[0]
        else:
            last_lr = None

    for epoch in range(args.epochs_start, args.epochs_start + args.epochs):
        epoch_start_time = time.perf_counter()

        if not args.weighted_sampler:
            errors = None

        train_loss, errors = train_one_epoch(
            args=args,
            model=model,
            model_ema=model_ema,
            accelerator=accelerator,
            criterion=criterion,
            dataloader=train_loader,
            optimizer=optimizer,
            errors=errors,
            epoch=epoch,
            print_freq=args.print_freq,
            logger=logger,
        )

        valid_loss, _ = evaluate(
            args,
            model=model,
            model_ema=model_ema,
            accelerator=accelerator,
            criterion=criterion,
            dataloader=val_loader,
            logger=logger,
        )

        update_val_result = best_metrics.update(valid_loss, epoch)

        accelerator.log({'train_loss': train_loss.metrics['total'].avg}, step=epoch)
        accelerator.log({'val_loss': valid_loss.metrics['total'].avg}, step=epoch)

        # Only main process should save model and compute validation statistics
        if accelerator.is_main_process:
            train_loss.log(
                logger,
                'train',
                epoch=epoch,
                time=time.perf_counter() - epoch_start_time,
                lr=optimizer.param_groups[0]['lr'],
            )
            valid_loss.log(logger, 'val', epoch=epoch)

            if update_val_result:
                save_checkpoint(
                    args, logger, accelerator, epoch, valid_loss, model, model_ema
                )

        if lr_scheduler is not None:
            lr_scheduler.step(metric=train_loss.metrics['total'].avg, epoch=epoch)

            if last_lr is not None and last_lr != lr_scheduler.get_last_lr()[0]:
                logger.log(
                    1,
                    f'Epoch [{epoch:>4}] -- New learning rate: {lr_scheduler.get_last_lr()[0]:.4g}',
                )

            last_lr = lr_scheduler.get_last_lr()[0]

    if test_loader is not None:
        # evaluate on the whole testing set
        test_loss, _ = evaluate(
            args,
            model=model,
            model_ema=model_ema,
            accelerator=accelerator,
            criterion=criterion,
            dataloader=test_loader,
            logger=logger,
        )

        accelerator.log({'test_loss': test_loss.metrics['total'].avg}, step=epoch)

        if accelerator.is_main_process:
            test_loss.log(logger, 'Test')


def _train(args):
    set_seeds(args.seed)
    set_dtype(args.dtype)

    if args.energy_weight == 0.0:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    else:
        ddp_kwargs = DistributedDataParallelKwargs()

    if args.wandb_project is not None:
        log_with = 'wandb'
    else:
        log_with = None

    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        log_with=log_with,
        kwargs_handlers=[ddp_kwargs],
    )

    try:
        _train_with_accelerator(args, accelerator)

    finally:
        accelerator.end_training()


def train(args):
    check_args_complete(args, 'train')

    if args.train_file is None:
        raise ArgumentError('--train-file is a required argument')
    if args.valid_file is None:
        raise ArgumentError('--valid-file is a required argument')
    if args.statistics_file is None:
        raise ArgumentError('--statistics-file is a required argument')
    if args.output_dir is None:
        raise ArgumentError('--output-dir is a required argument')
    if args.model is None:
        raise ArgumentError('--model is a required argument')

    if (
        args.energy_weight == 0.0
        and args.forces_weight == 0.0
        and args.stress_weight == 0.0
    ):
        raise ArgumentError('at least one non-zero loss weight is required')

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    _train(args)
