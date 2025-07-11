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
    check_args_consistency,
    get_loss_monitor,
)
from equitrain.checkpoint import load_checkpoint, save_checkpoint
from equitrain.data.loaders import dataloader_update_errors, get_dataloaders
from equitrain.evaluate import evaluate_main
from equitrain.logger import FileLogger
from equitrain.loss import LossCollection
from equitrain.loss_fn import LossFnCollection
from equitrain.loss_metrics import BestMetric, LossMetrics
from equitrain.model import get_model
from equitrain.model_freeze import model_freeze_params
from equitrain.train_optimizer import create_optimizer, update_weight_decay
from equitrain.train_scheduler import SchedulerWrapper, create_scheduler
from equitrain.utility import set_dtype, set_seeds

warnings.filterwarnings('ignore', message=r'.*TorchScript type system.*')


def fix_gradients(args, model: torch.nn.Module, accelerator: Accelerator):
    # Remove NaN and Inf from gradients
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0)

    # Clip gradients before optimization step
    if args.gradient_clipping is not None and args.gradient_clipping > 0:
        accelerator.clip_grad_value_(model.parameters(), args.gradient_clipping)


def forward_and_backward(
    model,
    data,
    data_list,
    i_,
    step,
    errors,
    loss_fn,
    loss_collection,
    accelerator,
    logger,
):
    y_pred = model(data)

    # Evaluate metric to be optimized
    loss, error = loss_fn(y_pred, data)

    try:
        # Backpropagate here to prevent out-of-memory errors, gradients
        # will be accumulated. Since we accumulate gradients over sub-batches,
        # we have to rescale before the backward pass
        accelerator.backward(loss.main['total'].value / len(data_list))

    except torch.OutOfMemoryError:
        logger.log(
            1,
            f'OOM error during backward pass on graph {i_} in batch {step}. Skipping...',
            force=True,
        )
        torch.cuda.empty_cache()

    loss_collection += loss

    errors[data.idx] = error


def train_one_epoch(
    args,
    model: torch.nn.Module,
    model_ema: ExponentialMovingAverage,
    accelerator: Accelerator,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    errors: torch.Tensor,
    epoch: int,
    logger: FileLogger,
):
    loss_fn = LossFnCollection(**vars(args))

    model.train()

    loss_metrics = LossMetrics(args)

    start_time = time.perf_counter()

    if errors is None:
        errors = torch.zeros(len(dataloader.dataset), device=accelerator.device)
    else:
        dataloader = dataloader_update_errors(
            args, dataloader, errors, accelerator, logger
        )

    if args.train_max_steps is None:
        total = len(dataloader)
    else:
        total = args.train_max_steps

    with tqdm(
        enumerate(dataloader),
        total=total,
        disable=not args.tqdm or not accelerator.is_main_process,
        desc='Training',
    ) as pbar:
        for step, data_list in pbar:
            # Compute a collection of loss metrics for monitoring purposes
            loss_collection = LossCollection(
                args.loss_monitor, device=accelerator.device
            )
            # Reset gradients
            optimizer.zero_grad()

            # Sub-batching causes deadlocks when the number of sub-batches varies between
            # processes. We need to loop over sub-batches withouth sync
            with accelerator.no_sync(model):
                for i_, data in enumerate(data_list):
                    # Break before last forward/backward pass
                    if i_ + 1 == len(data_list):
                        break

                    forward_and_backward(
                        model,
                        data,
                        data_list,
                        i_,
                        step,
                        errors,
                        loss_fn,
                        loss_collection,
                        accelerator,
                        logger,
                    )

            # Do final pass here to sync processes
            forward_and_backward(
                model,
                data,
                data_list,
                i_,
                step,
                errors,
                loss_fn,
                loss_collection,
                accelerator,
                logger,
            )

            # Handle NaN/Inf values and clip gradients
            fix_gradients(args, model, accelerator)
            # Perform gradient step on accumulated gradients
            optimizer.step()

            if model_ema is not None:
                model_ema.update()

            # Gather loss across processes for computing metrics
            loss_for_metrics = loss_collection.gather_for_metrics(accelerator)

            if not loss_for_metrics.main.isfinite():
                logger.log(2, 'NaN/Inf value detected during training')

            loss_metrics.update(loss_for_metrics)

            if accelerator.is_main_process:
                # Print intermediate performance statistics only for higher verbose levels
                if args.verbose > 1 and (
                    step % args.print_freq == 0 or step == len(dataloader) - 1
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
                        f'Training (lr={optimizer.param_groups[0]["lr"]:.0e}, loss={loss_metrics.main["total"].avg:.04g})'
                    )

            # Stop training if maximum number of steps is defined and reached
            if step >= total:
                break

    # Reset gradients
    optimizer.zero_grad()
    # Sum local errors across all processes
    accelerator.reduce(errors, reduction='sum')

    # Copy average weights to model
    if model_ema is not None:
        model_ema.copy_to(model.parameters())

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

    """ Network """
    model = get_model(args, logger=logger)

    """ Data Loader """
    train_loader, val_loader, test_loader = get_dataloaders(
        args, model.atomic_numbers, model.r_max, accelerator
    )

    """ Optimizer and LR Scheduler """
    optimizer = create_optimizer(args, model)
    lr_scheduler = create_scheduler(args, optimizer)

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
    if (result := load_checkpoint(args, model, model_ema, accelerator, logger))[0]:
        # Use epochs_start from the checkpoint
        args_checkpoint = result[1]
        check_args_consistency(args, args_checkpoint, logger)

    # Keep subset of parameters fixed during training
    model_freeze_params(args, model, logger=logger)

    # Allow users to modify the weight decay even when resuming from a checkpoint
    update_weight_decay(args, logger, optimizer)

    # Record the best validation loss and corresponding epoch
    best_metrics = BestMetric(args)

    """ Print training statistics """
    if accelerator.is_main_process:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.log(1, f'Number of params           : {n_parameters}')
        logger.log(1, f'Number of training points  : {len(train_loader.dataset)}')
        logger.log(
            1,
            f'Number of validation points: {len(val_loader.dataset) if val_loader is not None else 0}',
        )
        logger.log(
            1,
            f'Number of test points      : {len(test_loader.dataset) if test_loader is not None else 0}',
        )

    # Prediction errors for sampling training data accordingly
    if args.weighted_sampler:
        _, errors = evaluate_main(
            args,
            model=model,
            accelerator=accelerator,
            dataloader=train_loader,
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
        valid_loss, _ = evaluate_main(
            args,
            model=model,
            accelerator=accelerator,
            dataloader=val_loader,
            max_steps=args.valid_max_steps,
        )

        best_metrics.update(valid_loss.main, args.epochs_start - 1)

        accelerator.log(
            {'val_loss': valid_loss.main['total'].avg}, step=args.epochs_start - 1
        )

        if accelerator.is_main_process:
            valid_loss.log(logger, 'val', epoch=args.epochs_start - 1)

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
            dataloader=train_loader,
            optimizer=optimizer,
            errors=errors,
            epoch=epoch,
            logger=logger,
        )

        valid_loss, _ = evaluate_main(
            args,
            model=model,
            accelerator=accelerator,
            dataloader=val_loader,
            max_steps=args.valid_max_steps,
        )

        update_val_result = best_metrics.update(valid_loss.main, epoch)

        accelerator.log({'train_loss': train_loss.main['total'].avg}, step=epoch)
        accelerator.log({'val_loss': valid_loss.main['total'].avg}, step=epoch)
        accelerator.log({'lr': lr_scheduler.get_last_lr()[0]}, step=epoch)

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
                    args, epoch, valid_loss.main, model_ema, accelerator, logger
                )

        if lr_scheduler is not None:
            if args.scheduler_monitor == 'train':
                lr_scheduler.step(metric=train_loss.main['total'].avg, epoch=epoch)
            if args.scheduler_monitor == 'val':
                lr_scheduler.step(metric=valid_loss.main['total'].avg, epoch=epoch)

            if last_lr is not None and last_lr != lr_scheduler.get_last_lr()[0]:
                logger.log(
                    1,
                    f'Epoch [{epoch:>4}] -- New learning rate: {lr_scheduler.get_last_lr()[0]:.4g}',
                )

            last_lr = lr_scheduler.get_last_lr()[0]

    if test_loader is not None:
        # evaluate on the whole testing set
        test_loss, _ = evaluate_main(
            args,
            model=model,
            accelerator=accelerator,
            dataloader=test_loader,
        )

        accelerator.log({'test_loss': test_loss.main['total'].avg}, step=epoch)

        if accelerator.is_main_process:
            test_loss.log(logger, 'Test')


def _train(args):
    set_seeds(args.seed)
    set_dtype(args.dtype)

    if (
        args.energy_weight == 0.0
        or args.freeze_params is not None
        or args.unfreeze_params is not None
        or args.find_unused_parameters
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    else:
        ddp_kwargs = DistributedDataParallelKwargs()

    if args.wandb_project is not None:
        log_with = 'wandb'
    else:
        log_with = None

    # Filter out main loss type and convert to list
    args.loss_monitor = get_loss_monitor(args)

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

    if args.loss_type_energy is None:
        args.loss_type_energy = args.loss_type.lower()
    if args.loss_type_forces is None:
        args.loss_type_forces = args.loss_type.lower()
    if args.loss_type_stress is None:
        args.loss_type_stress = args.loss_type.lower()
    if (
        args.loss_type_energy != args.loss_type.lower()
        or args.loss_type_forces != args.loss_type.lower()
        or args.loss_type_stress != args.loss_type.lower()
    ):
        args.loss_type = 'mixed'

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    _train(args)
