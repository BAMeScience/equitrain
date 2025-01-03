import time
import torch
import os

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from pathlib import Path
from tqdm    import tqdm
from typing  import Iterable

from equitrain.argparser        import ArgumentError, ArgsFormatter, ArgsFilterSimple
from equitrain.data.loaders     import get_dataloaders
from equitrain.logger           import FileLogger
from equitrain.model            import get_model
from equitrain.loss             import GenericLossFn, Loss
from equitrain.utility          import set_dtype, set_seeds
from equitrain.train_checkpoint import load_checkpoint
from equitrain.train_optimizer  import create_optimizer
from equitrain.train_scheduler  import create_scheduler
from equitrain.train_metrics    import AverageMeter, LossMetric, BestMetric

import warnings
warnings.filterwarnings("ignore", message=r".*TorchScript type system.*")


def evaluate(args,
             model       : torch.nn.Module,
             accelerator : Accelerator,
             criterion   : torch.nn.Module,
             data_loader : Iterable,
    ):

    model.eval()
    criterion.eval()

    loss_metrics = LossMetric(args)

    for step, data_list in tqdm(enumerate(data_loader), total=len(data_loader), disable = not args.tqdm or not accelerator.is_main_process, desc="Evaluating"):

        loss_sum = Loss(device = accelerator.device)

        with accelerator.no_sync(model):

            for data in data_list:

                y_pred = model(data)

                loss = criterion(y_pred, data)

                if loss.isnan():
                    logger.log(1, f'Nan value detected. Skipping batch...')
                    continue

                loss_sum += loss

        # Gather loss across processes for computing metrics
        loss_for_metrics = loss_sum.gather_for_metrics(accelerator)

        # Check if loss was NaN for all iterations
        if loss_for_metrics.n == 0:
            continue

        loss_metrics.update(loss_for_metrics)

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

    loss_metrics = LossMetric(args)

    start_time = time.perf_counter()

    with tqdm(enumerate(data_loader), total=len(data_loader), disable = not args.tqdm or not accelerator.is_main_process, desc="Training") as pbar:

        for step, data_list in pbar:

            loss_sum = Loss(device = accelerator.device)

            # Sub-batching causes deadlocks when the number of sub-batches varies between
            # processes. We need to loop over sub-batches withouth sync
            with accelerator.no_sync(model):

                for i_, data in enumerate(data_list):

                    y_pred = model(data)

                    loss = criterion(y_pred, data)

                    if loss.isnan():
                        logger.log(1, f'Nan value detected. Skipping batch...')
                        continue

                    loss_sum += loss

            # Gather loss across processes for computing metrics
            loss_for_metrics = loss_sum.gather_for_metrics(accelerator)

            # Check if loss was NaN for all iterations
            if loss_for_metrics.n == 0:
                continue

            # Sync of models across processes occurs here
            optimizer.zero_grad()
            accelerator.backward(loss['total'])
            optimizer.step()

            loss_metrics.update(loss_for_metrics)

            if accelerator.is_main_process:

                # Print intermediate performance statistics only for higher verbose levels
                if i_ == 0 and args.verbose > 1:

                    if step % print_freq == 0 or step == len(data_loader) - 1:
                        w = time.perf_counter() - start_time
                        e = (step + 1) / len(data_loader)

                        loss_metrics.log_step(logger, epoch, step, len(data_loader), time = (1e3 * w / e / len(data_loader)), lr = optimizer.param_groups[0]["lr"])

                if args.tqdm:
                    pbar.set_description(f"Training (lr={optimizer.param_groups[0]['lr']:.0e}, loss={loss_metrics.metrics['total'].avg:.04f})")

    return loss_metrics


def _train_with_accelerator(args, accelerator: Accelerator):

    # Only main process should output information
    logger = FileLogger(log_to_file=True, enable_logging=accelerator.is_main_process, output_dir=args.output_dir, verbosity=args.verbose)
    logger.log(1, ArgsFormatter(args))

    ''' Data Loader '''
    train_loader, val_loader, test_loader = get_dataloaders(args, logger=logger)
    train_loader, val_loader, test_loader = accelerator.prepare(train_loader, val_loader, test_loader)

    ''' Network '''
    model = get_model(args, logger=logger)

    if accelerator.is_main_process:

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.log(1, 'Number of params: {}'.format(n_parameters))
    
    ''' Optimizer and LR Scheduler '''
    optimizer    = create_optimizer(args, model)
    lr_scheduler = create_scheduler(args, optimizer)
    last_lr      = None

    criterion = GenericLossFn(**vars(args))

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    load_checkpoint(args, logger, accelerator)

    # record the best validation loss and corresponding epoch
    best_metrics = BestMetric(args)

    if args.wandb_project is not None:
        accelerator.init_trackers(args.wandb_project, config=ArgsFilterSimple().filter(args))

    # Evaluate model before training
    if True:

        val_loss = evaluate(args, model=model, accelerator=accelerator, criterion=criterion, data_loader=val_loader)

        best_metrics.update(val_loss, args.epochs_start-1)

        accelerator.log({"val_loss": val_loss.metrics['total'].avg}, step=args.epochs_start-1)

        if accelerator.is_main_process:

            val_loss.log(logger, 'val', epoch = args.epochs_start-1)

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
        
        valid_loss = evaluate(args, model=model, accelerator=accelerator, criterion=criterion, data_loader=val_loader)

        update_val_result = best_metrics.update(valid_loss, epoch)

        accelerator.log({"train_loss": train_loss.metrics['total'].avg}, step=epoch)
        accelerator.log({  "val_loss": valid_loss.metrics['total'].avg}, step=epoch)

        # Only main process should save model and compute validation statistics
        if accelerator.is_main_process:

            train_loss.log(logger, 'train', epoch = epoch, time = time.perf_counter() - epoch_start_time, lr = optimizer.param_groups[0]["lr"])
            valid_loss.log(logger, 'val'  , epoch = epoch)

            if update_val_result:

                filename = 'best_val_epochs@{}_e@{:.4f}'.format(epoch, valid_loss.metrics['total'].avg)

                logger.log(1, f'Epoch [{epoch:>4}] -- Validation error decreased. Saving checkpoint to `{filename}`...')

                accelerator.save_state(
                    os.path.join(args.output_dir, filename),
                    safe_serialization=False)

        if lr_scheduler is not None:

            lr_scheduler.step(train_loss.metrics['total'].avg)

            if last_lr is not None and last_lr != lr_scheduler.get_last_lr()[0]:
                logger.log(1, f'Epoch [{epoch:>4}] -- New learning rate: {lr_scheduler.get_last_lr()[0]}')

            last_lr = lr_scheduler.get_last_lr()[0]

    if test_loader is not None:
        # evaluate on the whole testing set
        test_loss = evaluate(args, model=model, accelerator=accelerator, criterion=criterion, data_loader=test_loader)
 
        accelerator.log({"test_loss": test_loss.metrics['total'].avg}, step=epoch)

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
        log_with = "wandb"
    else:
        log_with = None

    accelerator = Accelerator(step_scheduler_with_optimizer = False, log_with = log_with, kwargs_handlers=[ddp_kwargs])

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

    if args.energy_weight == 0.0 and args.forces_weight == 0.0 and args.stress_weight == 0.0:
        raise ArgumentError("at least one non-zero loss weight is required")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    _train(args)
