from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass

import torch

try:  # pragma: no cover - optional new API
    from torch.func import functional_call as _functional_call
except ModuleNotFoundError:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call as _functional_call

from equitrain.backends.torch_wrappers import AbstractWrapper
from equitrain.finetune._lora_common import (
    effective_matrix_shape,
    resolve_rank,
    resolve_retained_fraction,
)

_SANITIZE_TOKEN = '__DOT__'
_LORA_A_SUFFIX = '__LORA_A__'
_LORA_B_SUFFIX = '__LORA_B__'


def _sanitize(name: str) -> str:
    return name.replace('.', _SANITIZE_TOKEN)


def _is_lora_param(name: str, param: torch.nn.Parameter) -> bool:
    return name.endswith('weight') and param.ndim >= 2


@dataclass(frozen=True)
class LoRASpec:
    name: str
    shape: tuple[int, ...]
    out_dim: int
    in_dim: int
    rank: int
    scale: float


class LoRAFineTuneWrapper(AbstractWrapper):
    """
    Wrap a torch model wrapper with LoRA adapters on eligible weight tensors.

    Eligible tensors are parameters named ``*.weight`` with ``ndim >= 2``.
    Higher-order weights are flattened to ``(shape[0], prod(shape[1:]))`` for
    the low-rank update and reshaped back to the original tensor shape.
    """

    def __init__(
        self,
        base_wrapper: AbstractWrapper,
        *,
        rank_fraction: float | int | None = None,
        rank_reduction: float | int | None = None,
        min_rank: int = 1,
        alpha: float | None = None,
    ):
        super().__init__(base_wrapper.model)
        self.base_wrapper = base_wrapper
        self.rank_fraction = rank_fraction
        self.rank_reduction = rank_reduction
        self.min_rank = int(min_rank)
        self.alpha = alpha

        retained_fraction = resolve_retained_fraction(
            rank_fraction=rank_fraction,
            rank_reduction=rank_reduction,
        )

        for param in self.base_wrapper.parameters():
            param.requires_grad_(False)

        self._lora_a_params = torch.nn.ParameterDict()
        self._lora_b_params = torch.nn.ParameterDict()
        self._lora_entries: list[
            tuple[
                str,
                torch.nn.Parameter,
                torch.nn.Parameter,
                torch.nn.Parameter,
                LoRASpec,
            ]
        ] = []
        self.lora_specs: dict[str, LoRASpec] = {}

        for name, param in self.base_wrapper.named_parameters():
            if not _is_lora_param(name, param):
                continue

            out_dim, in_dim = effective_matrix_shape(tuple(param.shape))
            rank = resolve_rank(
                min(out_dim, in_dim),
                retained_fraction=retained_fraction,
                min_rank=self.min_rank,
            )
            scale = 1.0 if alpha is None else float(alpha) / float(rank)
            spec = LoRASpec(
                name=name,
                shape=tuple(param.shape),
                out_dim=out_dim,
                in_dim=in_dim,
                rank=rank,
                scale=scale,
            )

            lora_a = torch.nn.Parameter(param.new_empty((rank, in_dim)))
            lora_b = torch.nn.Parameter(param.new_zeros((out_dim, rank)))
            torch.nn.init.normal_(lora_a, mean=0.0, std=1.0 / math.sqrt(in_dim))

            sanitized = _sanitize(name)
            self._lora_a_params[f'{sanitized}{_LORA_A_SUFFIX}'] = lora_a
            self._lora_b_params[f'{sanitized}{_LORA_B_SUFFIX}'] = lora_b
            self._lora_entries.append((name, param, lora_a, lora_b, spec))
            self.lora_specs[name] = spec

    def _merged_parameter(
        self,
        base_param: torch.nn.Parameter,
        lora_a: torch.nn.Parameter,
        lora_b: torch.nn.Parameter,
        spec: LoRASpec,
    ) -> torch.Tensor:
        delta = torch.matmul(lora_b, lora_a).reshape(spec.shape)
        return base_param + spec.scale * delta

    def lora_parameters(self) -> Iterator[torch.nn.Parameter]:
        for _, _, lora_a, lora_b, _ in self._lora_entries:
            yield lora_a
            yield lora_b

    def named_lora_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]:
        for name, _, lora_a, lora_b, _ in self._lora_entries:
            yield f'{name}.lora_a', lora_a
            yield f'{name}.lora_b', lora_b

    def forward(self, *args, **kwargs):
        lora_by_name = {
            name: (base_param, lora_a, lora_b, spec)
            for name, base_param, lora_a, lora_b, spec in self._lora_entries
        }
        params = OrderedDict()
        for name, param in self.base_wrapper.named_parameters():
            if name in lora_by_name:
                base_param, lora_a, lora_b, spec = lora_by_name[name]
                params[name] = self._merged_parameter(
                    base_param,
                    lora_a,
                    lora_b,
                    spec,
                )
            else:
                params[name] = param
        for name, buffer in self.base_wrapper.named_buffers():
            params[name] = buffer
        return _functional_call(self.base_wrapper, params, args, kwargs, strict=False)

    def export(self, *args, **kwargs):
        originals = [param.detach().clone() for _, param, _, _, _ in self._lora_entries]
        with torch.no_grad():
            for _, base_param, lora_a, lora_b, spec in self._lora_entries:
                base_param.copy_(
                    self._merged_parameter(base_param, lora_a, lora_b, spec)
                )
        try:
            if hasattr(self.base_wrapper, 'export'):
                return getattr(self.base_wrapper, 'export')(*args, **kwargs)
            if not args:
                raise TypeError('export() missing required filename argument.')
            return torch.save(self.model, args[0])
        finally:
            with torch.no_grad():
                for (_, base_param, _, _, _), original in zip(
                    self._lora_entries, originals
                ):
                    base_param.copy_(original)

    def __getattr__(self, item):
        if item in {
            'base_wrapper',
            'model',
            '_lora_a_params',
            '_lora_b_params',
            '_lora_entries',
            'lora_specs',
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


__all__ = ['LoRASpec', 'LoRAFineTuneWrapper']
