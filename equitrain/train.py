import logging
import time
import re
import torch
import os

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from pathlib import Path
from tqdm    import tqdm
from typing  import Iterable

from equitrain.argparser       import ArgumentError, ArgsFormatter, ArgsFilterSimple
from equitrain.data.loaders    import get_dataloaders
from equitrain.logger          import FileLogger
from equitrain.model           import get_model
from equitrain.loss            import GenericLoss
from equitrain.utility         import set_dtype, set_seeds
from equitrain.train_optimizer import create_optimizer
from equitrain.train_scheduler import create_scheduler
from equitrain.train_metrics   import AverageMeter, log_metrics, update_best_results

import warnings
warnings.filterwarnings("ignore", message=r".*TorchScript type system.*")


def evaluate(args,
             model       : torch.nn.Module,
             accelerator : Accelerator,
             criterion   : torch.nn.Module,
             data_loader : Iterable,
             max_iter    = -1
    ):

    model.eval()
    criterion.eval()

    loss_metrics = {
        'total' : AverageMeter(),
        'energy': AverageMeter(),
        'forces': AverageMeter(),
        'stress': AverageMeter(),
    }

    for step, data in tqdm(enumerate(data_loader), total=len(data_loader), disable = not args.tqdm or not accelerator.is_main_process, desc="Evaluating"):

        y_pred = model(data)

        loss = criterion(y_pred, data)

        loss_metrics['total'].update(loss['total'].item(), n=y_pred['energy'].shape[0])

        if y_pred['energy'] is not None:
            loss_metrics['energy'].update(loss['energy'].item(), n=y_pred['energy'].shape[0])
        if y_pred['forces'] is not None:
            loss_metrics['forces'].update(loss['forces'].item(), n=y_pred['forces'].shape[0])
        if y_pred['stress'] is not None:
            loss_metrics['stress'].update(loss['stress'].item(), n=y_pred['stress'].shape[0])

        if ((step + 1) >= max_iter) and (max_iter != -1):
            break

    return loss_metrics


def train_one_epoch(args, 
                    model       : torch.nn.Module,
                    accelerator : Accelerator,
                    criterion   : torch.nn.Module,
                    data_loader : Iterable,
                    optimizer   : torch.optim.Optimizer,
                    epoch       : int,
                    print_freq  : int = 100,
                    logger            = None,
    ):

    model.train()
    criterion.train()

    loss_metrics = {
        'total' : AverageMeter(),
        'energy': AverageMeter(),
        'forces': AverageMeter(),
        'stress': AverageMeter(),
    }

    start_time = time.perf_counter()

    with tqdm(enumerate(data_loader), total=len(data_loader), disable = not args.tqdm or not accelerator.is_main_process, desc="Training") as pbar:

        for step, data in pbar:

            # prevent out of memory error
            if args.batch_edge_limit > 0:
                if data.edge_index.shape[1] > args.batch_edge_limit:
                    logger.info(f'Batch edge limit violated. Batch has {data.edge_index.shape[1]} edges. Skipping batch...')
                    continue

            y_pred = model(data)

            loss = criterion(y_pred, data)

            if torch.isnan(loss['total']):
                logger.info(f'Nan value detected. Skipping batch...')
                continue

            optimizer.zero_grad()
            accelerator.backward(loss['total'])
            optimizer.step()

            loss_metrics['total'].update(loss['total'].item(), n=y_pred['energy'].shape[0])

            if args.energy_weight > 0.0:
                loss_metrics['energy'].update(loss['energy'].item(), n=y_pred['energy'].shape[0])
            if args.force_weight > 0.0:
                loss_metrics['forces'].update(loss['forces'].item(), n=y_pred['forces'].shape[0])
            if args.stress_weight > 0.0:
                loss_metrics['stress'].update(loss['stress'].item(), n=y_pred['stress'].shape[0])

            if accelerator.is_main_process:

                # Print intermediate performance statistics only for higher verbose levels
                if args.verbose > 2:

                    if step % print_freq == 0 or step == len(data_loader) - 1:
                        w = time.perf_counter() - start_time
                        e = (step + 1) / len(data_loader)

                        info_str_prefix  = 'Epoch [{epoch:>4}][{step:>6}/{length}] -- '.format(epoch=epoch, step=step, length=len(data_loader))
                        info_str_postfix = ', time/step={time_per_step:.0f}ms'.format(
                            time_per_step=(1e3 * w / e / len(data_loader))
                        )
                        info_str_postfix += ', lr={:.2e}'.format(optimizer.param_groups[0]["lr"])

                        log_metrics(args, logger, info_str_prefix, info_str_postfix, loss_metrics)

                if args.tqdm:
                    pbar.set_description(f"Training (lr={optimizer.param_groups[0]['lr']:.0e}, loss={loss_metrics['total'].avg:.04f})")


    return loss_metrics


def _train_with_accelerator(args, accelerator: Accelerator):

    # Only main process should output information
    logger = FileLogger(is_master=True, is_rank0=accelerator.is_main_process, output_dir=args.output_dir)
    if args.verbose > 0:
        logger.info(ArgsFormatter(args))

    ''' Data Loader '''
    train_loader, val_loader, test_loader = get_dataloaders(args, logger=logger)
    train_loader, val_loader, test_loader = accelerator.prepare(train_loader, val_loader, test_loader)

    ''' Network '''
    model = get_model(args, logger=logger)

    if accelerator.is_main_process:

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if args.verbose > 0:
            logger.info('Number of params: {}'.format(n_parameters))
    
    ''' Optimizer and LR Scheduler '''
    optimizer    = create_optimizer(args, model)
    lr_scheduler = create_scheduler(args, optimizer)

    criterion = GenericLoss(**vars(args))

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    if args.load_checkpoint is not None:

        if args.verbose > 0:
            logger.info(f'Loading checkpoint {args.load_checkpoint}...')

        accelerator.load_state(args.load_checkpoint)

    # Check and update epochs arguments
    if (m := re.match('.*best_[a-zA-Z]+_epochs@([0-9]+)_', args.load_checkpoint)) is not None:
        args.epochs_start = int(m[1])+1
    if args.epochs_start < 0:
        args.epochs_start = 1

    # record the best validation and testing loss and corresponding epochs
    best_metrics = {'val_epoch': 0, 'test_epoch': 0, 
         'val_energy_loss': float('inf'),  'val_forces_loss': float('inf'),  'val_stress_loss': float('inf'),
        'test_energy_loss': float('inf'), 'test_forces_loss': float('inf'), 'test_stress_loss': float('inf'),
    }

    if args.wandb_project is not None:
        accelerator.init_trackers(args.wandb_project, config=ArgsFilterSimple().filter(args))

    # Evaluate model before training
    if True:

        val_loss = evaluate(args, model=model, accelerator=accelerator, criterion=criterion, data_loader=val_loader)

        accelerator.log({"val_loss": val_loss['total'].avg}, step=args.epochs_start-1)

        if accelerator.is_main_process:

            # Print validation loss
            info_str_prefix  = 'Epoch [{epoch:>4}] Val   -- '.format(epoch=args.epochs_start-1)
            info_str_postfix = None

            log_metrics(args, logger, info_str_prefix, info_str_postfix, val_loss)

    for epoch in range(args.epochs_start, args.epochs_start+args.epochs):
        
        epoch_start_time = time.perf_counter()

        train_loss = train_one_epoch(
            args        = args,
            model       = model,
            accelerator = accelerator,
            criterion   = criterion,
            data_loader = train_loader,
            optimizer   = optimizer,
            epoch       = epoch,
            print_freq  = args.print_freq,
            logger      = logger)
        
        val_loss = evaluate(args, model=model, accelerator=accelerator, criterion=criterion, data_loader=val_loader)

        accelerator.log({"train_loss": train_loss['total'].avg}, step=epoch)
        accelerator.log({  "val_loss":   val_loss['total'].avg}, step=epoch)

        if lr_scheduler is not None:
            lr_scheduler.step(best_metrics['val_epoch'], epoch)

        # Only main process should save model and compute validation statistics
        if accelerator.is_main_process:

            update_val_result = update_best_results(criterion, best_metrics, val_loss, epoch)

            if update_val_result:

                filename = 'best_val_epochs@{}_e@{:.4f}'.format(epoch, val_loss['total'].avg)

                if args.verbose > 0:
                    logger.info(f'Validation error decreased. Saving checkpoint to `{filename}`...')

                accelerator.save_state(
                    os.path.join(args.output_dir, filename),
                    safe_serialization=False)

            info_str_prefix  = 'Epoch [{epoch:>4}] Train -- '.format(epoch=epoch)
            info_str_postfix = ', Time: {:.2f}s'.format(time.perf_counter() - epoch_start_time)

            log_metrics(args, logger, info_str_prefix, info_str_postfix, train_loss)

            info_str_prefix  = 'Epoch [{epoch:>4}] Val   -- '.format(epoch=epoch)
            info_str_postfix = None

            log_metrics(args, logger, info_str_prefix, info_str_postfix, val_loss)

    if test_loader is not None:
        # evaluate on the whole testing set
        test_loss = evaluate(args, model=model, accelerator=accelerator, criterion=criterion, data_loader=test_loader)
 
        accelerator.log({"test_loss": test_loss['total'].avg}, step=epoch)

        if accelerator.is_main_process:

            info_str_prefix  = 'Test -- '
            info_str_postfix = None

            log_metrics(args, logger, info_str_prefix, info_str_postfix, test_loss)


def _train(args):

    set_seeds(args.seed)
    set_dtype(args.dtype)

    if args.energy_weight == 0.0:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    else:
        ddp_kwargs = DistributedDataParallelKwargs()

    if args.wandb_project is not None:
        log_with = "wandb"
    else:
        log_with = None

    accelerator = Accelerator(log_with = log_with, kwargs_handlers=[ddp_kwargs])

    try:
        _train_with_accelerator(args, accelerator)

    finally:
        accelerator.end_training()


def train(args):

    if args.train_file is None:
        raise ArgumentError("--train-file is a required argument")
    if args.valid_file is None:
        raise ArgumentError("--valid-file is a required argument")
    if args.statistics_file is None:
        raise ArgumentError("--statistics-file is a required argument")
    if args.output_dir is None:
        raise ArgumentError("--output-dir is a required argument")
    if args.model is None:
        raise ArgumentError("--model is a required argument")

    if args.energy_weight == 0.0 and args.force_weight == 0.0 and args.stress_weight == 0.0:
        raise ArgumentError("at least one non-zero loss weight is required")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    _train(args)
