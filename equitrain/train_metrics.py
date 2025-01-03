
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


    def update(self, loss, n):
        """Update the loss metrics based on the current batch."""
        self.metrics['total'].update(loss['total'].item(), n=n)

        if self.metrics['energy'] is not None:
            self.metrics['energy'].update(loss['energy'].item(), n=n)

        if self.metrics['forces'] is not None:
            self.metrics['forces'].update(loss['forces'].item(), n=n)

        if self.metrics['stress'] is not None:
            self.metrics['stress'].update(loss['stress'].item(), n=n)


    def log(self, logger, prefix="", postfix=None):
        """Log the current loss metrics."""
        info_str = prefix
        info_str += f'loss: {self.metrics["total"].avg:.5f}'

        if self.metrics['energy'] is not None:
            info_str += f', loss_e: {self.metrics["energy"].avg:.5f}'

        if self.metrics['forces'] is not None:
            info_str += f', loss_f: {self.metrics["forces"].avg:.5f}'

        if self.metrics['stress'] is not None:
            info_str += f', loss_s: {self.metrics["stress"].avg:.5f}'

        if postfix is not None:
            info_str += postfix

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

    def update(self, metrics, epoch):
        """Update the best results if the current losses are better."""
        update_result = False

        loss_new = metrics['total'].avg
        loss_old = self.metrics['total']

        if loss_new < loss_old:

            self.metrics['total'] = metrics['total'].avg

            if self.metrics['energy'] is not None:
                self.metrics['energy'] = metrics['energy'].avg

            if self.metrics['forces'] is not None:
                self.metrics['forces'] = metrics['forces'].avg

            if self.metrics['stress'] is not None:
                self.metrics['stress'] = metrics['stress'].avg

            self.metrics['epoch'] = epoch
            update_result = True

        return update_result
