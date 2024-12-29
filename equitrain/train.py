import logging
import time
import torch
import math
import numpy as np
import os
import torch_geometric

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from pathlib import Path
from tqdm import tqdm
from typing  import Iterable, Optional

from torch_cluster import radius_graph

from equitrain.argparser       import ArgumentError, ArgsFormatter
from equitrain.data.loaders    import get_dataloaders
from equitrain.model           import get_model
from equitrain.loss            import GenericLoss
from equitrain.utility         import set_dtype, set_seeds
from equitrain.train_optimizer import create_optimizer
from equitrain.train_scheduler import create_scheduler

import warnings
warnings.filterwarnings("ignore", message=r".*TorchScript type system.*")


class FileLogger:
    def __init__(self, is_master=False, is_rank0=False, output_dir=None, logger_name='training'):
        # only call by master 
        # checked outside the class
        self.output_dir = output_dir
        if is_rank0:
            self.logger_name = logger_name
            self.logger = self.get_logger(output_dir, log_to_file=is_master)
        else:
            self.logger_name = None
            self.logger = NoOp()
        
        
    def get_logger(self, output_dir, log_to_file):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')

        if output_dir and log_to_file:
            
            time_formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(message)s')
            debuglog = logging.FileHandler(output_dir+'/debug.log')
            debuglog.setLevel(logging.DEBUG)
            debuglog.setFormatter(time_formatter)
            logger.addHandler(debuglog)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)
        
        # Reference: https://stackoverflow.com/questions/21127360/python-2-7-log-displayed-twice-when-logging-module-is-used-in-two-python-scri
        logger.propagate = False

        return logger

    def console(self, *args):
        self.logger.debug(*args)

    def event(self, *args):
        self.logger.warn(*args)

    def verbose(self, *args):
        self.logger.info(*args)

    def info(self, *args):
        self.logger.info(*args)


# no_op method/object that accept every signature
class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs): pass
        return no_op


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def compute_stats(data_loader, max_radius, logger, print_freq=1000):
    '''
        Compute mean of numbers of nodes and edges
    '''
    log_str = '\nCalculating statistics with '
    log_str = log_str + 'max_radius={}\n'.format(max_radius)
    logger.info(log_str)
        
    avg_node   = AverageMeter()
    avg_edge   = AverageMeter()
    avg_degree = AverageMeter()
    
    for step, data in enumerate(data_loader):
        
        pos   = data.pos
        batch = data.batch
        edge_src, edge_dst = radius_graph(
            pos,
            r=max_radius,
            batch=batch,
            max_num_neighbors=1000)

        batch_size = float(batch.max() + 1)
        num_nodes  = pos.shape[0]
        num_edges  = edge_src.shape[0]
        num_degree = torch_geometric.utils.degree(edge_src, num_nodes)
        num_degree = torch.sum(num_degree)
            
        avg_node  .update(num_nodes  / batch_size, batch_size)
        avg_edge  .update(num_edges  / batch_size, batch_size)
        avg_degree.update(num_degree / num_nodes, num_nodes)
            
        if step % print_freq == 0 or step == (len(data_loader) - 1):
            log_str = '[{}/{}]\tavg node: {}, '.format(step, len(data_loader), avg_node.avg)
            log_str += 'avg edge: {}, '.format(avg_edge.avg)
            log_str += 'avg degree: {}, '.format(avg_degree.avg)
            logger.info(log_str)


def log_metrics(args, logger, prefix, postfix, loss_metrics):

    info_str  = prefix
    info_str += 'loss: {loss:.5f}'.format(loss=loss_metrics['total'].avg)

    if args.energy_weight > 0.0:
        info_str += ', loss_e: {loss_e:.5f}'.format(
            loss_e=loss_metrics['energy'].avg,
        )
    if args.force_weight > 0.0:
        info_str += ', loss_f: {loss_f:.5f}'.format(
            loss_f=loss_metrics['forces'].avg,
        )
    if args.stress_weight > 0.0:
        info_str += ', loss_s: {loss_f:.5f}'.format(
            loss_f=loss_metrics['stress'].avg,
        )

    if postfix is not None:
        info_str += postfix

    logger.info(info_str)


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

    for step, data in tqdm(enumerate(data_loader), total=len(data_loader), disable = not args.tqdm or accelerator.process_index != 0, desc="Evaluating"):

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


def update_best_results(criterion, best_metrics, val_loss, epoch):

    update_result = False

    loss_new = criterion.compute_weighted_loss(
            val_loss['energy'].avg,
            val_loss['forces'].avg,
            val_loss['stress'].avg)
    loss_old = criterion.compute_weighted_loss(
            best_metrics['val_energy_loss'],
            best_metrics['val_forces_loss'],
            best_metrics['val_stress_loss'])

    if loss_new < loss_old:
        if criterion.energy_weight > 0.0:
            best_metrics['val_energy_loss'] = val_loss['energy'].avg
        if criterion.force_weight > 0.0:
            best_metrics['val_forces_loss'] = val_loss['forces'].avg
        if criterion.stress_weight > 0.0:
            best_metrics['val_stress_loss'] = val_loss['stress'].avg

        best_metrics['val_epoch'] = epoch

        update_result = True

    return update_result


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

    with tqdm(enumerate(data_loader), total=len(data_loader), disable = not args.tqdm or accelerator.process_index != 0, desc="Training") as pbar:

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

            if accelerator.process_index == 0:

                if args.verbose == 1:
                    # logging
                    if step % print_freq == 0 or step == len(data_loader) - 1:
                        w = time.perf_counter() - start_time
                        e = (step + 1) / len(data_loader)

                        info_str_prefix  = 'Epoch [{epoch:>4}][{step:>6}/{length}] -- '.format(epoch=epoch, step=step, length=len(data_loader))
                        info_str_postfix = ', time/step={time_per_step:.0f}ms'.format(
                            time_per_step=(1e3 * w / e / len(data_loader))
                        )
                        info_str_postfix += ', lr={:.2e}'.format(optimizer.param_groups[0]["lr"])

                        log_metrics(args, logger, info_str_prefix, info_str_postfix, loss_metrics)

                if args.verbose > 1:
                    pbar.set_description(f"Training (loss={loss_metrics['total'].avg:.04f})")


    return loss_metrics


def _train(args):

    set_seeds(args.seed)
    set_dtype(args.dtype)

    if args.energy_weight == 0.0:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    else:
        ddp_kwargs = DistributedDataParallelKwargs()

    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # Only main process should output information
    logger = FileLogger(is_master=True, is_rank0=(accelerator.process_index == 0), output_dir=args.output_dir)
    if args.verbose > 0:
        logger.info(ArgsFormatter(args))

    ''' Data Loader '''
    train_loader, val_loader, test_loader = get_dataloaders(args, logger=logger)
    train_loader, val_loader, test_loader = accelerator.prepare(train_loader, val_loader, test_loader)

    ''' Network '''
    model = get_model(args, logger=logger)

    if accelerator.process_index == 0:

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info('Number of params: {}'.format(n_parameters))
    
    ''' Optimizer and LR Scheduler '''
    optimizer    = create_optimizer(args, model)
    lr_scheduler = create_scheduler(args, optimizer)

    criterion = GenericLoss(**vars(args))

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    if args.load_checkpoint is not None:
        logger.info(f'Loading checkpoint {args.load_checkpoint}...')
        accelerator.load_state(args.load_checkpoint)
    
    # record the best validation and testing loss and corresponding epochs
    best_metrics = {'val_epoch': 0, 'test_epoch': 0, 
         'val_energy_loss': float('inf'),  'val_forces_loss': float('inf'),  'val_stress_loss': float('inf'),
        'test_energy_loss': float('inf'), 'test_forces_loss': float('inf'), 'test_stress_loss': float('inf'),
    }

    # Evaluate model before training
    if True:

        val_loss = evaluate(args, model=model, accelerator=accelerator, criterion=criterion, data_loader=val_loader)

        # Print validation loss
        info_str_prefix  = 'Epoch [{epoch:>4}] Val   -- '.format(epoch=0)
        info_str_postfix = None

        log_metrics(args, logger, info_str_prefix, info_str_postfix, val_loss)


    for epoch in range(1, args.epochs+1):
        
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

        if lr_scheduler is not None:
            lr_scheduler.step(best_metrics['val_epoch'], epoch)

        # Only main process should save model and compute validation statistics
        if accelerator.process_index == 0:

            update_val_result = update_best_results(criterion, best_metrics, val_loss, epoch)

            if update_val_result:

                filename = 'best_val_epochs@{}_e@{:.4f}'.format(epoch, val_loss['total'].avg)

                logger.info(f'Validation error decreased. Saving model to `{filename}`...')

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
 
        info_str_prefix  = 'Test -- '
        info_str_postfix = None

        log_metrics(args, logger, info_str_prefix, info_str_postfix, test_loss)


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
