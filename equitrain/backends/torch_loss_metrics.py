from __future__ import annotations

from equitrain.loss_metrics import AverageMeter

from .torch_loss import LossCollection


class LossMetric(dict):
    def __init__(self, args):
        self['total'] = AverageMeter()
        self['energy'] = AverageMeter() if args.energy_weight > 0.0 else None
        self['forces'] = AverageMeter() if args.forces_weight > 0.0 else None
        self['stress'] = AverageMeter() if args.stress_weight > 0.0 else None
        self['barrier'] = (
            AverageMeter() if getattr(args, 'barrier_weight', 0.0) > 0.0 else None
        )
        self['reaction_energy'] = (
            AverageMeter()
            if getattr(args, 'reaction_energy_weight', 0.0) > 0.0
            else None
        )
        self._weights = {
            'energy': float(getattr(args, 'energy_weight', 0.0)),
            'forces': float(getattr(args, 'forces_weight', 0.0)),
            'stress': float(getattr(args, 'stress_weight', 0.0)),
            'barrier': float(getattr(args, 'barrier_weight', 0.0)),
            'reaction_energy': float(getattr(args, 'reaction_energy_weight', 0.0)),
        }
        self._use_composite_total = (
            self['barrier'] is not None or self['reaction_energy'] is not None
        )

    def update(self, loss):
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
        if self['barrier'] is not None:
            self['barrier'].update(
                loss['barrier'].value.detach().item(),
                n=loss['barrier'].n.detach().item(),
            )
        if self['reaction_energy'] is not None:
            self['reaction_energy'].update(
                loss['reaction_energy'].value.detach().item(),
                n=loss['reaction_energy'].n.detach().item(),
            )
        if self._use_composite_total:
            self['total'].avg = self._composite_total()

    def _composite_total(self) -> float:
        total = 0.0
        for key, weight in self._weights.items():
            meter = self.get(key)
            if meter is not None and meter.count > 0:
                total += weight * meter.avg
        return total

    def log(
        self, logger, mode: str, epoch=None, step=None, time=None, lr=None, force=False
    ):
        if epoch is None:
            prefix = f'{mode}'
        else:
            prefix = f'Epoch [{epoch:>4}] -- {mode}'

        suffix = ''
        if time is not None:
            suffix += f', time: {time:.2f}s'
        if lr is not None:
            suffix += f', lr={lr:.2e}'

        message = f'{prefix}: {self["total"].avg:.5f}'
        if self['energy'] is not None:
            message += f', energy: {self["energy"].avg:.5f}'
        if self['forces'] is not None:
            message += f', forces: {self["forces"].avg:.5f}'
        if self['stress'] is not None:
            message += f', stress: {self["stress"].avg:.5f}'
        if self['barrier'] is not None:
            message += f', barrier: {self["barrier"].avg:.5f}'
        if self['reaction_energy'] is not None:
            message += f', reaction_energy: {self["reaction_energy"].avg:.5f}'
        message += suffix

        logger.log(1, message, force=force)

    def log_step(
        self, logger, epoch, step, length, mode, time=None, lr=None, force=False
    ):
        prefix = f'Epoch [{epoch:>4}][{step:>6}/{length}] -- {mode}'
        suffix = ''
        if time is not None:
            suffix += f', time: {time:.2f}s'
        if lr is not None:
            suffix += f', lr={lr:.2e}'

        message = f'{prefix}: {self["total"].avg:.6f}'
        if self['energy'] is not None:
            message += f', energy: {self["energy"].avg:.6f}'
        if self['forces'] is not None:
            message += f', forces: {self["forces"].avg:.6f}'
        if self['stress'] is not None:
            message += f', stress: {self["stress"].avg:.6f}'
        if self['barrier'] is not None:
            message += f', barrier: {self["barrier"].avg:.6f}'
        if self['reaction_energy'] is not None:
            message += f', reaction_energy: {self["reaction_energy"].avg:.6f}'
        message += suffix

        logger.log(1, message, force=force)


class BestMetric(dict):
    def __init__(self, args):
        self['total'] = float('inf')
        self['energy'] = float('inf') if args.energy_weight > 0.0 else None
        self['forces'] = float('inf') if args.forces_weight > 0.0 else None
        self['stress'] = float('inf') if args.stress_weight > 0.0 else None
        self['barrier'] = (
            float('inf') if getattr(args, 'barrier_weight', 0.0) > 0.0 else None
        )
        self['reaction_energy'] = (
            float('inf') if getattr(args, 'reaction_energy_weight', 0.0) > 0.0 else None
        )
        self['epoch'] = None

    def update(self, loss, epoch):
        update = False
        if loss['total'].avg < self['total']:
            self['total'] = loss['total'].avg
            if self['energy'] is not None:
                self['energy'] = loss['energy'].avg
            if self['forces'] is not None:
                self['forces'] = loss['forces'].avg
            if self['stress'] is not None:
                self['stress'] = loss['stress'].avg
            if self['barrier'] is not None:
                self['barrier'] = loss['barrier'].avg
            if self['reaction_energy'] is not None:
                self['reaction_energy'] = loss['reaction_energy'].avg
            self['epoch'] = epoch
            update = True
        return update


class LossMetrics(dict):
    def __init__(self, args):
        self.main = LossMetric(args)
        self.main_type = args.loss_type.lower()
        for loss_type in args.loss_monitor:
            self[loss_type] = LossMetric(args)

    def update(self, loss: LossCollection):
        self.main.update(loss.main)
        for loss_type, metric in self.items():
            metric.update(loss[loss_type])

    def log(
        self, logger, mode: str, epoch=None, step=None, time=None, lr=None, force=False
    ):
        self.main.log(
            logger,
            f'{mode:>5} {"[" + self.main_type + "]":7}',
            epoch=epoch,
            step=step,
            time=time,
            lr=lr,
        )
        for loss_type, metric in self.items():
            metric.log(
                logger,
                f'{mode:>5} {"[" + loss_type + "]":7}',
                epoch=epoch,
                step=step,
                time=None,
                lr=None,
                force=force,
            )

    def log_step(self, logger, epoch, step, length, time=None, lr=None, force=False):
        self.main.log_step(
            logger,
            epoch,
            step,
            length,
            f'{"[" + self.main_type + "]":7}',
            time=time,
            lr=lr,
        )
        for loss_type, metric in self.items():
            metric.log_step(
                logger,
                epoch,
                step,
                length,
                f'{"[" + loss_type + "]":7}',
                time=None,
                lr=None,
                force=force,
            )


__all__ = ['AverageMeter', 'LossMetric', 'BestMetric', 'LossMetrics']
