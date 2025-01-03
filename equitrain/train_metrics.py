
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


class LossMetric:

    def __init__(self, args):
        self.metrics = {
            'total' : AverageMeter(),
            'energy': AverageMeter() if args.energy_weight > 0.0 else None,
            'forces': AverageMeter() if args.forces_weight > 0.0 else None,
            'stress': AverageMeter() if args.stress_weight > 0.0 else None,
        }


    def update(self, loss):
        """Update the loss metrics based on the current batch."""
        self.metrics['total'].update(loss['total'].detach().item(), n = loss.n)

        if self.metrics['energy'] is not None:
            self.metrics['energy'].update(loss['energy'].detach().item(), n = loss.n)

        if self.metrics['forces'] is not None:
            self.metrics['forces'].update(loss['forces'].detach().item(), n = loss.n)

        if self.metrics['stress'] is not None:
            self.metrics['stress'].update(loss['stress'].detach().item(), n = loss.n)


    def log(self, logger, mode : str, epoch = None, step = None, time = None, lr = None):
        """Log the current loss metrics."""

        if epoch is None:
            info_str_prefix = f'{mode:>5} '
        else:
            info_str_prefix = f'Epoch [{epoch:>4}] -- {mode:>5} '

        info_str_postfix = ''

        if time is not None:
            info_str_postfix += f', time: {time:.2f}s'
        if lr is not None:
            info_str_postfix += f', lr={lr:.2e}'

        info_str  = info_str_prefix
        info_str += f'loss: {self.metrics["total"].avg:.5f}'

        if self.metrics['energy'] is not None:
            info_str += f', energy: {self.metrics["energy"].avg:.5f}'

        if self.metrics['forces'] is not None:
            info_str += f', forces: {self.metrics["forces"].avg:.5f}'

        if self.metrics['stress'] is not None:
            info_str += f', stress: {self.metrics["stress"].avg:.5f}'

        info_str += info_str_postfix

        logger.log(1, info_str)


    def log_step(self, logger, epoch, step, length, time = None, lr = None):
        """Log the current loss metrics."""

        info_str_prefix = f'Epoch [{epoch:>4}][{step:>6}/{length}] -- '
        info_str_postfix = ''

        if time is not None:
            info_str_postfix += f', time: {time:.2f}s'
        if lr is not None:
            info_str_postfix += f', lr={lr:.2e}'

        info_str  = info_str_prefix
        info_str += f'loss: {self.metrics["total"].avg:.5f}'

        if self.metrics['energy'] is not None:
            info_str += f', energy: {self.metrics["energy"].avg:.5f}'

        if self.metrics['forces'] is not None:
            info_str += f', forces: {self.metrics["forces"].avg:.5f}'

        if self.metrics['stress'] is not None:
            info_str += f', stress: {self.metrics["stress"].avg:.5f}'

        info_str += info_str_postfix

        logger.log(1, info_str)



class BestMetric:

    def __init__(self, args):

        self.metrics = {
            'total' : float('inf'),
            'energy': float('inf') if args.energy_weight > 0.0 else None,
            'forces': float('inf') if args.forces_weight > 0.0 else None,
            'stress': float('inf') if args.stress_weight > 0.0 else None,
            'epoch' : None,
        }


    def update(self, loss, epoch):
        """Update the best results if the current losses are better."""
        update_result = False

        loss_new = loss.metrics['total'].avg
        loss_old = self.metrics['total']

        if loss_new < loss_old:

            self.metrics['total'] = loss.metrics['total'].avg

            if self.metrics['energy'] is not None:
                self.metrics['energy'] = loss.metrics['energy'].avg

            if self.metrics['forces'] is not None:
                self.metrics['forces'] = loss.metrics['forces'].avg

            if self.metrics['stress'] is not None:
                self.metrics['stress'] = loss.metrics['stress'].avg

            self.metrics['epoch'] = epoch
            update_result = True

        return update_result
