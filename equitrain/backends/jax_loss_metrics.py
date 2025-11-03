from __future__ import annotations

import math

from equitrain.backends.jax_loss import JaxLossCollection
from equitrain.loss_metrics import AverageMeter


class LossMetric:
    def __init__(
        self, *, include_energy: bool, include_forces: bool, include_stress: bool
    ):
        self.meters = {
            'total': AverageMeter(),
            'energy': AverageMeter() if include_energy else None,
            'forces': AverageMeter() if include_forces else None,
            'stress': AverageMeter() if include_stress else None,
        }

    def update(self, collection: JaxLossCollection) -> None:
        component = collection.components['total']
        self.meters['total'].update(component.value, component.count)
        if self.meters['energy'] is not None:
            component = collection.components['energy']
            self.meters['energy'].update(component.value, component.count)
        if self.meters['forces'] is not None:
            component = collection.components['forces']
            self.meters['forces'].update(component.value, component.count)
        if self.meters['stress'] is not None:
            component = collection.components['stress']
            self.meters['stress'].update(component.value, component.count)

    def as_dict(self) -> dict[str, float | None]:
        result = {'total': self.meters['total'].avg}
        if self.meters['energy'] is not None:
            result['energy'] = self.meters['energy'].avg
        if self.meters['forces'] is not None:
            result['forces'] = self.meters['forces'].avg
        if self.meters['stress'] is not None:
            result['stress'] = self.meters['stress'].avg
        return result

    def log(
        self,
        logger,
        mode: str,
        *,
        epoch=None,
        step=None,
        length=None,
        time=None,
        lr=None,
        force=False,
    ) -> None:
        if epoch is None and step is None:
            prefix = mode
        elif step is None:
            prefix = f'Epoch [{epoch:>4}] -- {mode}'
        else:
            prefix = f'Epoch [{epoch:>4}][{step:>6}/{length}] -- {mode}'

        suffix = ''
        if time is not None:
            suffix += f', time: {time:.2f}s'
        if lr is not None:
            suffix += f', lr={lr:.2e}'

        message = f'{prefix}: total={self.meters["total"].avg:.6f}'
        if self.meters['energy'] is not None:
            message += f', energy={self.meters["energy"].avg:.6f}'
        if self.meters['forces'] is not None:
            message += f', forces={self.meters["forces"].avg:.6f}'
        if self.meters['stress'] is not None:
            message += f', stress={self.meters["stress"].avg:.6f}'
        message += suffix

        logger.log(1, message, force=force)


class LossMetrics:
    def __init__(
        self,
        *,
        include_energy: bool,
        include_forces: bool,
        include_stress: bool,
        loss_label: str,
    ):
        self.loss_label = loss_label
        self.main = LossMetric(
            include_energy=include_energy,
            include_forces=include_forces,
            include_stress=include_stress,
        )

    def update(self, collection: JaxLossCollection) -> None:
        self.main.update(collection)

    def as_dict(self) -> dict[str, float | None]:
        return self.main.as_dict()

    def log(
        self, logger, mode: str, *, epoch=None, time=None, lr=None, force=False
    ) -> None:
        label = f'{mode:>5} [{self.loss_label}]'
        self.main.log(logger, label, epoch=epoch, time=time, lr=lr, force=force)

    def log_step(
        self, logger, *, epoch, step, length, time=None, lr=None, force=False
    ) -> None:
        label = f'[{self.loss_label}]'
        self.main.log(
            logger,
            label,
            epoch=epoch,
            step=step,
            length=length,
            time=time,
            lr=lr,
            force=force,
        )


__all__ = ['AverageMeter', 'LossMetric', 'LossMetrics']
