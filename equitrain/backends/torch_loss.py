from __future__ import annotations

import torch


class LossComponent:
    def __init__(self, value: torch.Tensor = None, n: torch.Tensor = None, device=None):
        self.value = value if value is not None else torch.tensor(0.0, device=device)
        self.n = n if n is not None else torch.tensor(0.0, device=device)

    def __iadd__(self, component: LossComponent):
        self.value = (self.value * self.n + component.value * component.n) / (
            self.n + component.n
        )
        self.n += component.n
        return self

    def detach(self):
        result = LossComponent()
        result.value = self.value.detach()
        result.n = self.n.detach()
        return result

    def gather_for_metrics(self, accelerator):
        result = LossComponent(device=accelerator.device)

        values = accelerator.gather_for_metrics(self.value.detach())
        ns = accelerator.gather_for_metrics(self.n.detach())

        if len(values.shape) == 0:
            result += LossComponent(value=values, n=ns)
        else:
            for value, count in zip(values, ns, strict=True):
                result += LossComponent(value=value, n=count)

        return result


class Loss(dict):
    def __init__(self, device=None):
        self['total'] = LossComponent(device=device)
        self['energy'] = LossComponent(device=device)
        self['forces'] = LossComponent(device=device)
        self['stress'] = LossComponent(device=device)

    def __iadd__(self, loss: Loss):
        for key, component in loss.items():
            self[key] += component
        return self

    def isfinite(self):
        return torch.isfinite(self['total'].value)

    def detach(self):
        result = Loss()
        for key, value in self.items():
            result[key] = value.detach()
        return result

    def gather_for_metrics(self, accelerator):
        result = Loss(device=accelerator.device)
        for key, component in self.items():
            result[key] = component.gather_for_metrics(accelerator)
        return result


class LossCollection(dict):
    def __init__(self, loss_types, device=None):
        self.main = Loss(device=device)
        for loss_type in loss_types:
            self[loss_type] = Loss(device=device)

    def __iadd__(self, loss_collection: LossCollection):
        self.main += loss_collection.main
        for loss_type, loss in loss_collection.items():
            self[loss_type] += loss
        return self

    def gather_for_metrics(self, accelerator):
        result = LossCollection(list(self.keys()), device=accelerator.device)
        result.main = self.main.gather_for_metrics(accelerator)
        for loss_type, loss in self.items():
            result[loss_type] = loss.gather_for_metrics(accelerator)
        return result


__all__ = ['LossComponent', 'Loss', 'LossCollection']
