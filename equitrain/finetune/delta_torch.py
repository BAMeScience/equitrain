from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Iterator

import torch

try:  # pragma: no cover - optional new API
    from torch.func import functional_call as _functional_call
except ModuleNotFoundError:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call as _functional_call

from equitrain.backends.torch_wrappers import AbstractWrapper
from equitrain.finetune._torch_common import (
    clone_non_complete_tensor_storage,
    remap_legacy_model_prefix,
)

_SANITIZE_TOKEN = '__DOT__'
_RANGE_RE = re.compile(r'^(\d*)\s*[-:]\s*(\d*)$')


def _sanitize(name: str) -> str:
    return name.replace('.', _SANITIZE_TOKEN)


def _delta_layer_name(parameter_name: str) -> str:
    parts = parameter_name.split('.')
    if parts and parts[0] == 'model':
        parts = parts[1:]
    if not parts:
        return parameter_name

    top_level = parts[0]
    if top_level in {'interactions', 'products'} and len(parts) > 1:
        return f'{top_level}.{parts[1]}'
    if top_level == 'readouts':
        return top_level
    return top_level


def _parse_freeze_layers(freeze_layers, layer_names: tuple[str, ...]) -> set[int]:
    if freeze_layers is None or freeze_layers == '':
        return set()

    if isinstance(freeze_layers, int):
        tokens = [str(freeze_layers)]
    elif isinstance(freeze_layers, str):
        tokens = [token.strip() for token in freeze_layers.split(',') if token.strip()]
    else:
        tokens = []
        for item in freeze_layers:
            if isinstance(item, str):
                tokens.extend(
                    token.strip() for token in item.split(',') if token.strip()
                )
            else:
                tokens.append(str(item))

    selected: set[int] = set()
    max_index = len(layer_names) - 1
    for token in tokens:
        match = _RANGE_RE.fullmatch(token)
        if match is not None:
            start_s, end_s = match.groups()
            if start_s == '' and end_s == '':
                raise ValueError('Empty delta layer range is not valid.')
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else max_index
            if start > max_index:
                selected.add(start)
                continue
            if start > end:
                raise ValueError(f'Invalid delta layer range {token!r}.')
            selected.update(range(start, end + 1))
            continue

        try:
            selected.add(int(token))
        except ValueError as exc:
            raise ValueError(
                f'Invalid delta layer selector {token!r}; use e.g. "2-", "1,3", or "2-4".'
            ) from exc

    invalid = sorted(index for index in selected if index < 0 or index > max_index)
    if invalid:
        layers = ', '.join(f'{index}:{name}' for index, name in enumerate(layer_names))
        raise ValueError(
            f'Delta layer selector out of range: {invalid}. Available layers: {layers}'
        )
    return selected


class DeltaFineTuneWrapper(AbstractWrapper):
    """
    Wrap a :class:`~equitrain.backends.torch_wrappers.AbstractWrapper` instance with
    additive (delta) parameters that are trained while the original parameters remain
    frozen.

    The wrapper keeps a reference to the underlying base wrapper and proxies all
    attribute access to it. During the forward pass, the deltas are temporarily added
    to the base parameters.
    """

    def __init__(self, base_wrapper: AbstractWrapper, *, freeze_layers=None):
        # The base wrapper is the sole registered owner of the underlying model.
        torch.nn.Module.__init__(self)
        self.base_wrapper = base_wrapper
        self.freeze_layers = freeze_layers

        # Freeze original parameters.
        for param in self.base_wrapper.parameters():
            param.requires_grad_(False)

        # Create trainable delta parameters mirroring the base parameters.
        self._delta_params = torch.nn.ParameterDict()
        self._delta_entries: list[
            tuple[str, torch.nn.Parameter, torch.nn.Parameter]
        ] = []
        for name, param in self.base_wrapper.named_parameters():
            delta = torch.nn.Parameter(torch.zeros_like(param))
            sanitized = _sanitize(name)
            self._delta_params[sanitized] = delta
            self._delta_entries.append((name, param, delta))
        self._delta_layer_names = self._infer_delta_layer_names()
        self._delta_layer_by_name = {
            name: self._delta_layer_names.index(_delta_layer_name(name))
            for name, _, _ in self._delta_entries
        }
        self.freeze_delta_layers(freeze_layers)

    def _infer_delta_layer_names(self) -> tuple[str, ...]:
        layer_names = []
        for name, _, _ in self._delta_entries:
            layer_name = _delta_layer_name(name)
            if layer_name not in layer_names:
                layer_names.append(layer_name)
        return tuple(layer_names)

    # --------------------------------------------------------------------- helpers
    def delta_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Iterate over all delta parameters."""
        return iter(self._delta_params.values())

    def named_delta_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]:
        """Iterate over all named delta parameters using original parameter names."""
        for name, _, delta in self._delta_entries:
            yield name, delta

    @property
    def delta_layer_names(self) -> tuple[str, ...]:
        """Return semantic delta layer names in optimizer traversal order."""
        return self._delta_layer_names

    def freeze_delta_layers(self, freeze_layers=None) -> None:
        """Freeze selected semantic delta layers; ``None`` keeps all deltas trainable."""
        self.freeze_layers = freeze_layers
        selected = _parse_freeze_layers(freeze_layers, self._delta_layer_names)
        for name, _, delta in self._delta_entries:
            delta.requires_grad_(self._delta_layer_by_name[name] not in selected)

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

    def get_fine_tune_export_config(self):
        config = {'wrapper': 'delta'}
        if self.freeze_layers is not None:
            config['freeze_layers'] = self.freeze_layers
        return config

    @property
    def model(self):
        return self.base_wrapper.model

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        clone_non_complete_tensor_storage(state_dict)
        return state_dict

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        remap_legacy_model_prefix(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # --------------------------------------------------------------------- proxies
    def __getattr__(self, item):
        if item in {
            'base_wrapper',
            '_delta_params',
            '_delta_entries',
            '_delta_layer_names',
            '_delta_layer_by_name',
        }:
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
