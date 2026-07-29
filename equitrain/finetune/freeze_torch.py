from __future__ import annotations

from collections.abc import Iterator

import torch

from equitrain.backends.torch_wrappers import AbstractWrapper
from equitrain.finetune._layer_selection import (
    infer_semantic_layer_names,
    parse_layer_selection,
    semantic_layer_indices,
)
from equitrain.finetune._torch_common import (
    clone_non_complete_tensor_storage,
    remap_legacy_model_prefix,
)


class FreezeFineTuneWrapper(AbstractWrapper):
    """
    Wrap a torch model wrapper and fine-tune selected base parameters directly.

    Unlike delta or LoRA fine-tuning, this wrapper does not add adapter weights.
    It only controls which base parameters have gradients enabled, so export can
    write the updated base model without a merge step.
    """

    def __init__(self, base_wrapper: AbstractWrapper, *, freeze_layers=None):
        torch.nn.Module.__init__(self)
        self.base_wrapper = base_wrapper
        self.freeze_layers = freeze_layers

        self._fine_tune_entries = list(self.base_wrapper.named_parameters())
        parameter_names = [name for name, _ in self._fine_tune_entries]
        self._freeze_layer_names = infer_semantic_layer_names(parameter_names)
        self._freeze_layer_by_name = semantic_layer_indices(
            parameter_names,
            self._freeze_layer_names,
        )
        self.freeze_model_layers(freeze_layers)

    @property
    def freeze_layer_names(self) -> tuple[str, ...]:
        """Return semantic layer names in optimizer traversal order."""
        return self._freeze_layer_names

    def freeze_model_layers(self, freeze_layers=None) -> None:
        """Freeze selected semantic layers; ``None`` keeps all layers trainable."""
        self.freeze_layers = freeze_layers
        selected = parse_layer_selection(freeze_layers, self._freeze_layer_names)
        for name, param in self._fine_tune_entries:
            param.requires_grad_(self._freeze_layer_by_name[name] not in selected)

    def fine_tune_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Iterate over currently trainable base parameters."""
        for _, param in self._fine_tune_entries:
            if param.requires_grad:
                yield param

    def named_fine_tune_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]:
        """Iterate over currently trainable base parameters with original names."""
        for name, param in self._fine_tune_entries:
            if param.requires_grad:
                yield name, param

    def forward(self, *args, **kwargs):
        return self.base_wrapper(*args, **kwargs)

    def export(self, *args, **kwargs):
        if hasattr(self.base_wrapper, 'export'):
            return getattr(self.base_wrapper, 'export')(*args, **kwargs)
        if not args:
            raise TypeError('export() missing required filename argument.')
        return torch.save(self.model, args[0])

    def get_fine_tune_export_config(self):
        config = {'wrapper': 'freeze'}
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

    def __getattr__(self, item):
        if item in {
            'base_wrapper',
            '_fine_tune_entries',
            '_freeze_layer_names',
            '_freeze_layer_by_name',
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


__all__ = ['FreezeFineTuneWrapper']
