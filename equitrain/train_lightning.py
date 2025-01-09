import logging
import math
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim import SGD, Adadelta, Adam, AdamW, RMSprop

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
    """Computes and stores the average and current value"""

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


def add_weight_decay_and_groups(
    model, weight_decay=1e-5, filter_bias_and_bn=True, lr_groups=None
):
    decay = []
    no_decay = []

    # Parameters can be grouped according to a specific key, if lr_groups is provided
    param_groups = {}  # Dictionary to store groups based on user-defined criteria

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen parameters

        # Apply filtering for biases, batch normalization, affine weights, and affine biases
        if filter_bias_and_bn and (
            name.endswith('.bias')
            or 'bn' in name.lower()
            or name.endswith('.affine_weight')
            or name.endswith('.affine_bias')
            or 'bias.' in name
            or 'mean_shift' in name
        ):
            no_decay.append(param)
        else:
            decay.append(param)

        # If user provides a grouping criterion, group parameters accordingly
        if lr_groups:
            for group_name, group_fn in lr_groups.items():
                if group_fn(name):
                    if group_name not in param_groups:
                        param_groups[group_name] = {'params': []}
                    param_groups[group_name]['params'].append(param)

    # Build the final parameter group list
    param_group_list = [
        {'params': decay, 'weight_decay': weight_decay},  # Parameters with weight decay
        {
            'params': no_decay,
            'weight_decay': 0.0,
        },  # Parameters without weight decay (biases, BN layers, etc.)
    ]

    # Include user-defined parameter groups
    for group_name, group_params in param_groups.items():
        param_group_list.append(group_params)

    return param_group_list


def compute_weighted_loss(args, energy_loss, force_loss, stress_loss):
    result = 0.0
    # handle initial values correctly when weights are zero, i.e. 0.0*Inf -> NaN
    if energy_loss is not None and (
        not math.isinf(energy_loss) or args.energy_weight > 0.0
    ):
        result += args.energy_weight * energy_loss
    if force_loss is not None and (
        not math.isinf(force_loss) or args.forces_weight > 0.0
    ):
        result += args.forces_weight * force_loss
    if stress_loss is not None and (
        not math.isinf(stress_loss) or args.stress_weight > 0.0
    ):
        result += args.stress_weight * stress_loss

    return result


def create_optimizer(args, model, filter_bias_and_bn=True):
    # Example: User-defined learning rate groups based on layer names
    lr_groups = {
        'group1': lambda name: 'layer1'
        in name,  # Apply specific criterion for grouping
        'group2': lambda name: 'layer2' in name,
    }

    # Add weight decay and group parameters based on bias/BN filtering and user-defined groups
    params = add_weight_decay_and_groups(
        model,
        weight_decay=args.weight_decay,
        filter_bias_and_bn=filter_bias_and_bn,
        lr_groups=lr_groups,
    )

    # Create the optimizer based on user input
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
        raise ValueError(f'Unsupported optimizer: {args.opt}')


class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs):
            pass

        return no_op


class FileLogger:
    def __init__(
        self,
        is_master=False,
        is_rank0=False,
        output_dir=None,
        logger_name='training',
        version='1',
    ):
        # only call by master
        # checked outside the class
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
            time_formatter = logging.Formatter(
                '%(asctime)s - %(filename)s:%(lineno)d - %(message)s'
            )
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

    def console(self, *args):
        self.logger.debug(*args)

    def event(self, *args):
        self.logger.warn(*args)

    def verbose(self, *args):
        self.logger.info(*args)

    def info(self, *args):
        self.logger.info(*args)

    def finalize(self, status):
        self.logger.info(f'Training finalized with status: {status}')

    def log_graph(self, model):
        pass

    def save(self):
        pass

    def log_metrics(self, metrics, step=None):
        for key, value in metrics.items():
            if step is not None:
                self.logger.info(f'{key}: {value} at step {step}')
            else:
                self.logger.info(f'{key}: {value}')

    def after_save_checkpoint(self, checkpoint):
        self.logger.info(f'Checkpoint saved at {checkpoint}')


class EquiTrainModule(pl.LightningModule):
    def __init__(self, args, model, criterion, logger=None):
        super().__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.custom_logger = logger

        self.train_loss_metrics = {
            'total': AverageMeter(),
            'energy': AverageMeter(),
            'forces': AverageMeter(),
            'stress': AverageMeter(),
        }
        self.val_loss_metrics = {
            'total': AverageMeter(),
            'energy': AverageMeter(),
            'forces': AverageMeter(),
            'stress': AverageMeter(),
        }

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        e_true = batch.y
        f_true = batch['force']
        s_true = batch['stress']

        e_pred, f_pred, s_pred = self(batch)

        loss_e, loss_f, loss_s = None, None, None
        if self.args.energy_weight > 0.0:
            loss_e = self.criterion(e_pred, e_true)
        if self.args.forces_weight > 0.0:
            loss_f = self.criterion(f_pred, f_true)
        if self.args.stress_weight > 0.0:
            loss_s = self.criterion(s_pred, s_true)

        loss = compute_weighted_loss(self.args, loss_e, loss_f, loss_s)
        if torch.isnan(loss):
            self.logger.info('NaN value detected. Skipping batch...')
            return None

        # Log the metrics into PyTorch Lightning’s system
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        if loss_e is not None:
            self.log(
                'loss_e',
                loss_e,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch.num_graphs,
            )
        if loss_f is not None:
            self.log(
                'loss_f',
                loss_f,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch.num_graphs,
            )
        if loss_s is not None:
            self.log(
                'loss_s',
                loss_s,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch.num_graphs,
            )

        # Update the training metrics
        self.train_loss_metrics['total'].update(loss.item(), n=e_pred.shape[0])
        if self.args.energy_weight > 0.0:
            self.train_loss_metrics['energy'].update(loss_e.item(), n=e_pred.shape[0])
        if self.args.forces_weight > 0.0:
            self.train_loss_metrics['forces'].update(loss_f.item(), n=f_pred.shape[0])
        if self.args.stress_weight > 0.0:
            self.train_loss_metrics['stress'].update(loss_s.item(), n=s_pred.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        e_true = batch.y
        f_true = batch['force']
        s_true = batch['stress']

        pred_e, pred_f, pred_s = self(batch)

        loss_e, loss_f, loss_s = None, None, None
        if pred_e is not None:
            loss_e = self.criterion(pred_e, e_true)
        if pred_f is not None:
            loss_f = self.criterion(pred_f, f_true)
        if pred_s is not None:
            loss_s = self.criterion(pred_s, s_true)

        loss = compute_weighted_loss(self.args, loss_e, loss_f, loss_s)

        # Update the validation metrics
        self.val_loss_metrics['total'].update(loss.item(), n=pred_e.shape[0])
        if self.args.energy_weight > 0.0:
            self.val_loss_metrics['energy'].update(loss_e.item(), n=pred_e.shape[0])
        if self.args.forces_weight > 0.0:
            self.val_loss_metrics['forces'].update(loss_f.item(), n=pred_e.shape[0])
        if self.args.stress_weight > 0.0:
            self.val_loss_metrics['stress'].update(loss_s.item(), n=pred_s.shape[0])

        # Log validation loss (this will be used by ModelCheckpoint)
        self.log('val_loss', loss, prog_bar=True, batch_size=batch.num_graphs)

        if loss_e is not None:
            self.log('loss_e', loss_e, batch_size=batch.num_graphs)
        if loss_f is not None:
            self.log('loss_f', loss_f, batch_size=batch.num_graphs)
        if loss_s is not None:
            self.log('loss_s', loss_s, batch_size=batch.num_graphs)

        return loss

    def configure_optimizers(self):
        optimizer = create_optimizer(self.args, self.model)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=self.args.patience_epochs
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'val_loss',  # The metric to monitor for PlateauLR
                'interval': 'epoch',
                'frequency': 1,
                'reduce_on_plateau': True,  # Tell Lightning it's a PlateauLR type scheduler
            },
        }

    def on_train_epoch_end(self):
        # Use the custom logger to log metrics at the end of the training epoch
        log_metrics(
            self.args,
            self.custom_logger,
            f'Epoch [{self.current_epoch}] Train -- ',
            None,
            self.train_loss_metrics,
        )
        self.reset_train_loss_metrics()

    def on_validation_epoch_end(self):
        # Use the custom logger to log metrics at the end of the validation epoch
        log_metrics(
            self.args,
            self.custom_logger,
            f'Epoch [{self.current_epoch}] Val -- ',
            None,
            self.val_loss_metrics,
        )
        self.reset_val_loss_metrics()

    def reset_train_loss_metrics(self):
        for meter in self.train_loss_metrics.values():
            meter.reset()

    def reset_val_loss_metrics(self):
        for meter in self.val_loss_metrics.values():
            meter.reset()


def _train(args):
    logger = FileLogger(is_master=True, is_rank0=True, output_dir=args.output_dir)
    logger.info(args)

    torch.set_float32_matmul_precision('medium')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.batch_size = 32  # Reducing batch size to increase the number of batches

    # ! unused
    # Distributed Data Parallel options
    # ddp_kwargs = (
    #     DistributedDataParallelKwargs(find_unused_parameters=True)
    #     if args.energy_weight == 0.0
    #     else DistributedDataParallelKwargs()
    # )

    # ! unused
    # accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    """ Data Loader """
    train_loader, val_loader, test_loader, r_max = get_dataloaders(args, logger=logger)

    """ Network """
    model = get_model(
        r_max,
        args,
        compute_force=args.forces_weight > 0.0,
        compute_stress=args.stress_weight > 0.0,
        logger=logger,
    )

    # Create Criterion
    criterion = torch.nn.L1Loss()

    """ Lightning Module """
    lightning_model = EquiTrainModule(args, model, criterion, logger)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',  # This is the key you are logging
        save_top_k=1,  # Save the best checkpoint
        mode='min',  # We're minimizing the loss
        filename='best-checkpoint',
    )

    """ PyTorch Lightning Trainer """
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1,
        accelerator='auto',
        precision=16 if args.mixed_precision else 32,
        log_every_n_steps=1,
        logger=logger,  # Integrate your custom logger
        default_root_dir=args.output_dir,
        callbacks=[checkpoint_callback],
    )

    """ Start Training """
    trainer.fit(lightning_model, train_loader, val_loader)

    # Test on the test dataset
    if test_loader is not None:
        trainer.test(lightning_model, test_loader)


def train_lightning(args):
    if args.train_file is None:
        raise ValueError('--train-file is a required argument')
    if args.valid_file is None:
        raise ValueError('--valid-file is a required argument')
    if args.statistics_file is None:
        raise ValueError('--statistics-file is a required argument')
    if args.output_dir is None:
        raise ValueError('--output-dir is a required argument')

    if (
        args.energy_weight == 0.0
        and args.forces_weight == 0.0
        and args.stress_weight == 0.0
    ):
        raise ValueError('at least one non-zero loss weight is required')

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    _train(args)
