from equitrain.loss import LossCollection


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossMetric(dict):
    def __init__(self, args):
        self['total'] = AverageMeter()
        self['energy'] = AverageMeter() if args.energy_weight > 0.0 else None
        self['forces'] = AverageMeter() if args.forces_weight > 0.0 else None
        self['stress'] = AverageMeter() if args.stress_weight > 0.0 else None

    def update(self, loss):
        """Update the loss metrics based on the current batch."""
        self['total'].update(
            loss['total'].value.detach().item(), n=loss['total'].n.detach().item()
        )

        if self['energy'] is not None:
            self['energy'].update(
                loss['energy'].value.detach().item(), n=loss['energy'].n.detach().item()
            )

        if self['forces'] is not None:
            self['forces'].update(
                loss['forces'].value.detach().item(), n=loss['forces'].n.detach().item()
            )

        if self['stress'] is not None:
            self['stress'].update(
                loss['stress'].value.detach().item(), n=loss['stress'].n.detach().item()
            )

    def log(self, logger, mode: str, epoch=None, step=None, time=None, lr=None):
        """Log the current loss metrics."""

        if epoch is None:
            info_str_prefix = f'{mode}'
        else:
            info_str_prefix = f'Epoch [{epoch:>4}] -- {mode}'

        info_str_postfix = ''

        if time is not None:
            info_str_postfix += f', time: {time:.2f}s'
        if lr is not None:
            info_str_postfix += f', lr={lr:.2e}'

        info_str = info_str_prefix
        info_str += f': {self["total"].avg:.5f}'

        if self['energy'] is not None:
            info_str += f', energy: {self["energy"].avg:.5f}'

        if self['forces'] is not None:
            info_str += f', forces: {self["forces"].avg:.5f}'

        if self['stress'] is not None:
            info_str += f', stress: {self["stress"].avg:.5f}'

        info_str += info_str_postfix

        logger.log(1, info_str)

    def log_step(self, logger, epoch, step, length, mode, time=None, lr=None):
        """Log the current loss metrics."""

        info_str_prefix = f'Epoch [{epoch:>4}][{step:>6}/{length}] -- {mode}'
        info_str_postfix = ''

        if time is not None:
            info_str_postfix += f', time: {time:.2f}s'
        if lr is not None:
            info_str_postfix += f', lr={lr:.2e}'

        info_str = info_str_prefix
        info_str += f': {self["total"].avg:.6f}'

        if self['energy'] is not None:
            info_str += f', energy: {self["energy"].avg:.6f}'

        if self['forces'] is not None:
            info_str += f', forces: {self["forces"].avg:.6f}'

        if self['stress'] is not None:
            info_str += f', stress: {self["stress"].avg:.6f}'

        info_str += info_str_postfix

        logger.log(1, info_str)


class BestMetric(dict):
    def __init__(self, args):
        self['total'] = float('inf')
        self['energy'] = float('inf') if args.energy_weight > 0.0 else None
        self['forces'] = float('inf') if args.forces_weight > 0.0 else None
        self['stress'] = float('inf') if args.stress_weight > 0.0 else None
        self['epoch'] = None

    def update(self, loss, epoch):
        """Update the best results if the current losses are better."""
        update_result = False

        loss_new = loss['total'].avg
        loss_old = self['total']

        if loss_new < loss_old:
            self['total'] = loss['total'].avg

            if self['energy'] is not None:
                self['energy'] = loss['energy'].avg

            if self['forces'] is not None:
                self['forces'] = loss['forces'].avg

            if self['stress'] is not None:
                self['stress'] = loss['stress'].avg

            self['epoch'] = epoch
            update_result = True

        return update_result


class LossMetrics(dict):
    def __init__(self, args):
        self.main = LossMetric(args)
        self.main_type = args.loss_type
        for loss_type in args.loss_monitor.split(','):
            self[loss_type] = LossMetric(args)

    def update(self, loss: LossCollection):
        self.main.update(loss.main)
        for loss_type, loss_metric in self.items():
            self[loss_type].update(loss[loss_type])

    def log(self, logger, mode: str, epoch=None, step=None, time=None, lr=None):
        self.main.log(
            logger,
            f'{mode:>5} {"[" + self.main_type + "]":7}',
            epoch=epoch,
            step=step,
            time=time,
            lr=lr,
        )
        for loss_type, loss_metric in self.items():
            loss_metric.log(
                logger,
                f'{mode:>5} {"[" + loss_type + "]":7}',
                epoch=epoch,
                step=step,
                time=None,
                lr=None,
            )

    def log_step(self, logger, epoch, step, length, time=None, lr=None):
        self.main.log_step(
            logger,
            epoch,
            step,
            length,
            f'{"[" + self.main_type + "]":7}',
            time=time,
            lr=lr,
        )
        for loss_type, loss_metric in self.items():
            loss_metric.log_step(
                logger,
                epoch,
                step,
                length,
                f'{"[" + loss_type + "]":7}',
                time=None,
                lr=None,
            )
