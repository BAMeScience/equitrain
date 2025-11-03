from __future__ import annotations

import jax
import numpy as np

from equitrain.backends.jax_runtime import ensure_multiprocessing_spawn

_LOSS_KEYS = ('total', 'energy', 'forces', 'stress')


ensure_multiprocessing_spawn()


class LossComponent:
    def __init__(self):
        self.value = 0.0
        self.count = 0.0

    def update(self, value: float, count: float) -> None:
        if count <= 0.0:
            return
        if self.count == 0.0:
            self.value = float(value)
            self.count = float(count)
            return
        total = self.value * self.count + float(value) * float(count)
        self.count += float(count)
        self.value = total / self.count


class JaxLossCollection:
    def __init__(self) -> None:
        self.components = {key: LossComponent() for key in _LOSS_KEYS}

    def update_component(self, name: str, value: float, count: float) -> None:
        if name in self.components:
            self.components[name].update(value, count)

    def update_from_metrics(self, metrics: dict[str, tuple[float, float]]) -> None:
        for key, (value, count) in metrics.items():
            self.update_component(key, float(value), float(count))

    def mean(self) -> float:
        component = self.components['total']
        return component.value if component.count else float('nan')

    def as_dict(self) -> dict[str, float]:
        return {key: component.value for key, component in self.components.items()}


def update_collection_from_aux(collection: JaxLossCollection, aux) -> np.ndarray:
    aux_host = jax.device_get(aux)
    metrics = aux_host['metrics']
    for key, (value, count) in metrics.items():
        collection.update_component(key, float(value), float(count))
    return np.asarray(aux_host['per_graph_error'])


__all__ = ['LossComponent', 'JaxLossCollection', 'update_collection_from_aux']
