import torch
import lightning as L
import math
import logging
import numpy as np

from pathlib import Path
from torch.optim import SGD, Adam, AdamW, RMSprop, Adadelta

from equitrain.data.loaders import get_dataloaders
from equitrain.model import get_model


def log_metrics(args, logger, prefix, postfix, loss_metrics):
    info_str = prefix
    info_str += 'loss: {loss:.5f}'.format(loss=loss_metrics['total'].avg)
    if args.energy_weight > 0.0:
        info_str += ', loss_e: {loss_e:.5f}'.format(
            loss_e=loss_metrics['energy'].avg,
        )
    if args.forces_weight > 0.0:
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


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def add_weight_decay_and_groups(model, weight_decay=1e-5, filter_bias_and_bn=True, lr_groups=None):
    decay = []
    no_decay = []
    param_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if filter_bias_and_bn and (
                name.endswith(".bias") or "bn" in name.lower() or name.endswith(".affine_weight") or
                name.endswith(".affine_bias") or 'bias.' in name or 'mean_shift' in name):
            no_decay.append(param)
        else:
            decay.append(param)
        if lr_groups:
            for group_name, group_fn in lr_groups.items():
                if group_fn(name):
                    if group_name not in param_groups:
                        param_groups[group_name] = {'params': []}
                    param_groups[group_name]['params'].append(param)

    param_group_list = [{'params': decay, 'weight_decay': weight_decay},
                        {'params': no_decay, 'weight_decay': 0.0}]
    for group_name, group_params in param_groups.items():
        param_group_list.append(group_params)
    return param_group_list


def compute_weighted_loss(args, energy_loss, force_loss, stress_loss):
    result = 0.0
    if energy_loss is not None and (not math.isinf(energy_loss) or args.energy_weight > 0.0):
        result += args.energy_weight * energy_loss
    if force_loss is not None and (not math.isinf(force_loss) or args.forces_weight > 0.0):
        result += args.forces_weight * force_loss
    if stress_loss is not None and (not math.isinf(stress_loss) or args.stress_weight > 0.0):
        result += args.stress_weight * stress_loss
    return result


def create_optimizer(args, model, filter_bias_and_bn=True):
    lr_groups = {
        'group1': lambda name: 'layer1' in name,
        'group2': lambda name: 'layer2' in name,
    }
    params = add_weight_decay_and_groups(model, weight_decay=args.weight_decay, filter_bias_and_bn=filter_bias_and_bn,
                                         lr_groups=lr_groups)

    if args.opt == 'sgd':
        return SGD(params, lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        return Adam(params, lr=args.lr)
    elif args.opt == 'adamw':
        return AdamW(params, lr=args.lr)
    elif args.opt == 'rmsprop':
        return RMSprop(params, lr=args.lr, alpha=0.99)
    elif args.opt == 'adadelta':
        return Adadelta(params, lr=args.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {args.opt}")


class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs): pass
        return no_op


class FileLogger:
    def __init__(self, is_master=False, is_rank0=False, output_dir=None, logger_name='training', version='1'):
        self.output_dir = output_dir
        self.save_dir = output_dir
        self.name = logger_name
        self.version = version
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
            debuglog = logging.FileHandler(output_dir + '/debug.log')
            debuglog.setLevel(logging.DEBUG)
            debuglog.setFormatter(time_formatter)
            logger.addHandler(debuglog)
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)
        logger.propagate = False
        return logger

    def info(self, *args):
        self.logger.info(*args)


class EquiTrainModule:
    def __init__(self, args, model, criterion, logger=None):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.custom_logger = logger
        self.train_loss_metrics = {'total': AverageMeter(), 'energy': AverageMeter(), 'forces': AverageMeter(),
                                   'stress': AverageMeter()}
        self.val_loss_metrics = {'total': AverageMeter(), 'energy': AverageMeter(), 'forces': AverageMeter(),
                                 'stress': AverageMeter()}

    def forward(self, data):
        return self.model(data)


def _train(args):
    logger = FileLogger(is_master=True, is_rank0=True, output_dir=args.output_dir)
    logger.info(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize Lightning Fabric
    fabric = L.Fabric(accelerator="cuda", devices=1, strategy="ddp", precision="16-mixed")

    # Launch the Fabric environment for distributed training
    fabric.launch()

    ''' Data Loader '''
    train_loader, val_loader, test_loader, r_max = get_dataloaders(args, logger=logger)

    ''' Network '''
    model = get_model(r_max, args, compute_force=args.forces_weight > 0.0, compute_stress=args.stress_weight > 0.0,
                      logger=logger)

    # Create Criterion
    criterion = torch.nn.L1Loss()

    # Create Optimizer
    optimizer = create_optimizer(args, model)

    # Setup model, optimizer, and dataloaders using Fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    ''' Start Training '''
    for epoch in range(args.epochs):
        logger.info(f"Epoch nº {epoch+1}")
        logger.info("-----------------")


        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            e_true = batch.y
            f_true = batch['force']
            s_true = batch['stress']

            e_pred, f_pred, s_pred = model(batch)
            loss_e, loss_f, loss_s = None, None, None
            if args.energy_weight > 0.0:
                loss_e = criterion(e_pred, e_true)
            if args.forces_weight > 0.0:
                loss_f = criterion(f_pred, f_true)
            if args.stress_weight > 0.0:
                loss_s = criterion(s_pred, s_true)

            loss = compute_weighted_loss(args, loss_e, loss_f, loss_s)
            logger.info(f"Training loss: {loss.item()}")

            fabric.backward(loss)
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                e_true = batch.y
                f_true = batch['force']
                s_true = batch['stress']

                e_pred, f_pred, s_pred = model(batch)
                loss_e, loss_f, loss_s = None, None, None
                if args.energy_weight > 0.0:
                    loss_e = criterion(e_pred, e_true)
                if args.forces_weight > 0.0:
                    loss_f = criterion(f_pred, f_true)
                if args.stress_weight > 0.0:
                    loss_s = criterion(s_pred, s_true)

                val_loss = compute_weighted_loss(args, loss_e, loss_f, loss_s)
                logger.info(f"Validation loss: {val_loss.item()}\n")


def train_fabric(args):
    if args.train_file is None:
        raise ValueError("--train-file is a required argument")
    if args.valid_file is None:
        raise ValueError("--valid-file is a required argument")
    if args.statistics_file is None:
        raise ValueError("--statistics-file is a required argument")
    if args.output_dir is None:
        raise ValueError("--output-dir is a required argument")
    if args.energy_weight == 0.0 and args.forces_weight == 0.0 and args.stress_weight == 0.0:
        raise ValueError("at least one non-zero loss weight is required")
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    _train(args)
