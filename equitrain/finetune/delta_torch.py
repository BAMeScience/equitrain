from __future__ import annotations

from collections import OrderedDict
from typing import Iterator, Tuple

import torch

try:  # pragma: no cover - optional new API
    from torch.func import functional_call as _functional_call
except ModuleNotFoundError:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call as _functional_call

from equitrain.backends.torch_wrappers import AbstractWrapper

_SANITIZE_TOKEN = '__DOT__'


def _sanitize(name: str) -> str:
    return name.replace('.', _SANITIZE_TOKEN)


class DeltaFineTuneWrapper(AbstractWrapper):
    """
    Wrap a :class:`~equitrain.backends.torch_wrappers.AbstractWrapper` instance with
    additive (delta) parameters that are trained while the original parameters remain
    frozen.

    The wrapper keeps a reference to the underlying base wrapper and proxies all
    attribute access to it. During the forward pass, the deltas are temporarily added
    to the base parameters.
    """

    def __init__(self, base_wrapper: AbstractWrapper):
        super().__init__(base_wrapper.model)
        self.base_wrapper = base_wrapper

        # Freeze original parameters.
        for param in self.base_wrapper.parameters():
            param.requires_grad_(False)

        # Create trainable delta parameters mirroring the base parameters.
        self._delta_params = torch.nn.ParameterDict()
        self._delta_entries: list[
            Tuple[str, torch.nn.Parameter, torch.nn.Parameter]
        ] = []
        for name, param in self.base_wrapper.named_parameters():
            delta = torch.nn.Parameter(torch.zeros_like(param))
            sanitized = _sanitize(name)
            self._delta_params[sanitized] = delta
            self._delta_entries.append((name, param, delta))

    # --------------------------------------------------------------------- helpers
    def delta_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Iterate over the trainable delta parameters."""
        return iter(self._delta_params.values())

    def named_delta_parameters(self) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """Iterate over the named delta parameters using the original parameter names."""
        for name, _, delta in self._delta_entries:
            yield name, delta

    # --------------------------------------------------------------------- overrides
    def forward(self, *args, **kwargs):
        params = OrderedDict(
            (name, base_param + delta)
            for name, base_param, delta in self._delta_entries
        )
        for name, buffer in self.base_wrapper.named_buffers():
            params[name] = buffer
        return _functional_call(self.base_wrapper, params, args, kwargs, strict=False)

    def export(self, *args, **kwargs):
        originals = [param.detach().clone() for _, param, _ in self._delta_entries]
        with torch.no_grad():
            for _, base_param, delta in self._delta_entries:
                base_param.add_(delta)
        try:
            if hasattr(self.base_wrapper, 'export'):
                return getattr(self.base_wrapper, 'export')(*args, **kwargs)
            if not args:
                raise TypeError('export() missing required filename argument.')
            return torch.save(self.model, args[0])
        finally:
            with torch.no_grad():
                for (_, base_param, _), original in zip(self._delta_entries, originals):
                    base_param.copy_(original)

    # --------------------------------------------------------------------- proxies
    def __getattr__(self, item):
        if item in {'base_wrapper', 'model', '_delta_params', '_delta_entries'}:
            return super().__getattr__(item)
        return getattr(self.base_wrapper, item)

    @property
    def atomic_numbers(self):
        return self.base_wrapper.atomic_numbers

    @property
    def atomic_energies(self):
        return self.base_wrapper.atomic_energies

    @property
    def r_max(self):
        return self.base_wrapper.r_max

    @r_max.setter
    def r_max(self, value):
        self.base_wrapper.r_max = value


__all__ = ['DeltaFineTuneWrapper']
