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
from typing  import Iterable, Optional

from torch_cluster import radius_graph

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP
from timm.optim.adabelief import AdaBelief
from timm.scheduler import create_scheduler

from equitrain.argparser   import ArgumentError
from equitrain.dataloaders import get_dataloaders
from equitrain.model       import get_model, ModelWrapper

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


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (name.endswith(".bias") or name.endswith(".affine_weight")  
            or name.endswith(".affine_bias") or name.endswith('.mean_shift')
            or 'bias.' in name 
            or name in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def optimizer_kwargs(cfg):
    """ cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict(
        optimizer_name=cfg.opt,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum)
    if getattr(cfg, 'opt_eps', None) is not None:
        kwargs['eps'] = cfg.opt_eps
    if getattr(cfg, 'opt_betas', None) is not None:
        kwargs['betas'] = cfg.opt_betas
    if getattr(cfg, 'opt_args', None) is not None:
        kwargs.update(cfg.opt_args)
    return kwargs


def create_optimizer(args, model, filter_bias_and_bn=True):
    """ Legacy optimizer factory for backwards compatibility.
    NOTE: Use create_optimizer_v2 for new code.
    """
    return create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        filter_bias_and_bn=filter_bias_and_bn,
    )


def create_optimizer_v2(
        model: torch.nn.Module,
        optimizer_name: str = 'sgd',
        lr: Optional[float] = None,
        weight_decay: float = 0.,
        momentum: float = 0.9,
        filter_bias_and_bn: bool = True,
        **kwargs):
    """ Create an optimizer.

    TODO currently the model is passed in and all parameters are selected for optimization.
    For more general use an interface that allows selection of parameters to optimize and lr groups, one of:
      * a filter fn interface that further breaks params into groups in a weight_decay compatible fashion
      * expose the parameters interface and leave it up to caller

    Args:
        model (nn.Module): model containing parameters to optimize
        optimizer_name: name of optimizer to create
        lr: initial learning rate
        weight_decay: weight decay to apply in optimizer
        momentum:  momentum for momentum based optimizers (others may use betas via kwargs)
        filter_bias_and_bn:  filter out bias, bn and other 1d params from weight decay
        **kwargs: extra optimizer specific kwargs to pass through

    Returns:
        Optimizer
    """
    opt_lower = optimizer_name.lower()
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()
    #if 'fused' in opt_lower:
    #    assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=lr, weight_decay=weight_decay, **kwargs)
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = torch.optim.SGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = torch.optim.SGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = torch.optim.Adam(parameters, **opt_args) 
    elif opt_lower == 'adabelief':
        optimizer = AdaBelief(parameters, rectify=False, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = torch.optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':        
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = torch.optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = torch.optim.RMSprop(parameters, alpha=0.9, momentum=momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=momentum, **opt_args)
    else:
        assert False and "Invalid optimizer"

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer


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


def compute_weighted_loss(args, energy_loss, force_loss, stress_loss):
    result = 0.0
    # handle initial values correctly when weights are zero, i.e. 0.0*Inf -> NaN
    if energy_loss is not None and (not math.isinf(energy_loss) or args.energy_weight > 0.0):
        result += args.energy_weight * energy_loss
    if force_loss is not None and (not math.isinf(force_loss) or args.force_weight > 0.0):
        result += args.force_weight * force_loss
    if stress_loss is not None and (not math.isinf(stress_loss) or args.stress_weight > 0.0):
        result += args.stress_weight * stress_loss

    return result


def evaluate(args,
             model      : torch.nn.Module,
             criterion  : torch.nn.Module,
             data_loader: Iterable, 
             max_iter   = -1):

    model.eval()
    criterion.eval()

    loss_metrics = {
        'total' : AverageMeter(),
        'energy': AverageMeter(),
        'forces': AverageMeter(),
        'stress': AverageMeter(),
    }

    for step, data in enumerate(data_loader):

        pred_e, pred_f, pred_s = model(data)

        loss_e = None
        loss_f = None
        loss_s = None

        if pred_e is not None:
            loss_e = criterion(pred_e, data.y)
        if pred_f is not None:
            loss_f = criterion(pred_f, data['force'])
        if pred_s is not None:
            loss_s = criterion(pred_s, data['stress'])

        loss = compute_weighted_loss(args, loss_e, loss_f, loss_s)

        loss_metrics['total'].update(loss.item(), n=pred_e.shape[0])

        if pred_e is not None:
            loss_metrics['energy'].update(loss_e.item(), n=pred_e.shape[0])
        if pred_f is not None:
            loss_metrics['forces'].update(loss_f.item(), n=pred_f.shape[0])
        if pred_s is not None:
            loss_metrics['stress'].update(loss_s.item(), n=pred_s.shape[0])

        if ((step + 1) >= max_iter) and (max_iter != -1):
            break

    return loss_metrics


def update_best_results(args, best_metrics, val_loss, epoch):

    update_result = False

    loss_new = compute_weighted_loss(args,
            val_loss['energy'].avg,
            val_loss['forces'].avg,
            val_loss['stress'].avg)
    loss_old = compute_weighted_loss(args,
            best_metrics['val_energy_loss'],
            best_metrics['val_forces_loss'],
            best_metrics['val_stress_loss'])

    if loss_new < loss_old:
        if args.energy_weight > 0.0:
            best_metrics['val_energy_loss'] = val_loss['energy'].avg
        if args.force_weight > 0.0:
            best_metrics['val_forces_loss'] = val_loss['forces'].avg
        if args.stress_weight > 0.0:
            best_metrics['val_stress_loss'] = val_loss['stress'].avg

        best_metrics['val_epoch'] = epoch

        update_result = True

    return update_result


def train_one_epoch(args, 
                    model: torch.nn.Module, accelerator: Accelerator, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, 
                    print_freq: int = 100, 
                    logger=None):

    model.train()
    criterion.train()

    loss_metrics = {
        'total' : AverageMeter(),
        'energy': AverageMeter(),
        'forces': AverageMeter(),
        'stress': AverageMeter(),
    }

    start_time = time.perf_counter()

    for step, data in enumerate(data_loader):

        # prevent out of memory error
        if args.batch_edge_limit > 0:
            if data.edge_index.shape[1] > args.batch_edge_limit:
                logger.info(f'Batch edge limit violated. Batch has {data.edge_index.shape[1]} edges. Skipping batch...')
                continue

        e_true = data.y
        f_true = data['force']
        s_true = data['stress']

        e_pred, f_pred, s_pred = model(data)

        loss_e = None
        loss_f = None
        loss_s = None

        if args.energy_weight > 0.0:
            loss_e = criterion(e_pred, e_true)
        if args.force_weight > 0.0:
            loss_f = criterion(f_pred, f_true)
        if args.stress_weight > 0.0:
            loss_s = criterion(s_pred, s_true)

        loss = compute_weighted_loss(args, loss_e, loss_f, loss_s)

        if torch.isnan(loss):
            logger.info(f'Nan value detected. Skipping batch...')
            continue

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        loss_metrics['total'].update(loss.item(), n=e_pred.shape[0])

        if args.energy_weight > 0.0:
            loss_metrics['energy'].update(loss_e.item(), n=e_pred.shape[0])
        if args.force_weight > 0.0:
            loss_metrics['forces'].update(loss_f.item(), n=f_pred.shape[0])
        if args.stress_weight > 0.0:
            loss_metrics['stress'].update(loss_s.item(), n=s_pred.shape[0])

        if accelerator.process_index == 0:

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

    return loss_metrics


def _train(args):
    
    logger = FileLogger(is_master=True, is_rank0=True, output_dir=args.output_dir)
    logger.info(args)
    
    # since dataset needs random 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # set dtype globally
    torch.set_default_dtype(torch.float64)

    if args.energy_weight == 0.0:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    else:
        ddp_kwargs = DistributedDataParallelKwargs()

    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    ''' Data Loader '''
    train_loader, val_loader, test_loader, r_max = get_dataloaders(args, logger=logger)
    train_loader, val_loader, test_loader = accelerator.prepare(train_loader, val_loader, test_loader)

    ''' Network '''
    model = get_model(r_max, args,
        compute_force=args.force_weight > 0.0,
        compute_stress=args.stress_weight > 0.0,
        logger=logger)

    if accelerator.process_index == 0:

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info('Number of params: {}'.format(n_parameters))
    
    ''' Optimizer and LR Scheduler '''
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    if args.warmup_epochs == 0:
        if hasattr(lr_scheduler, 'warmup_t'):
            # manually disable warmup (timm bugfix)
            lr_scheduler.warmup_t = -1

    criterion = torch.nn.L1Loss() 

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

        val_loss = evaluate(args, model=model, criterion=criterion, data_loader=val_loader)

        # Print validation loss
        if accelerator.process_index == 0:

            info_str_prefix  = 'Epoch [{epoch:>4}] Val   -- '.format(epoch=0)
            info_str_postfix = None

            log_metrics(args, logger, info_str_prefix, info_str_postfix, val_loss)


    for epoch in range(1, args.epochs+1):
        
        epoch_start_time = time.perf_counter()

        train_loss = train_one_epoch(
            args=args,
            model=model,
            accelerator=accelerator,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            print_freq=args.print_freq,
            logger=logger)
        
        val_loss = evaluate(args, model=model, criterion=criterion, data_loader=val_loader)

        lr_scheduler.step(best_metrics['val_epoch'], epoch)

        # Only main process should save model
        if accelerator.process_index == 0:

            update_val_result = update_best_results(args, best_metrics, val_loss, epoch)

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
        test_loss = evaluate(args, model=model, criterion=criterion, data_loader=test_loader)
 
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
