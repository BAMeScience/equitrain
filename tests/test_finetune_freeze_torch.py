from __future__ import annotations

import pytest
import torch

from equitrain.argparser import ArgsFormatter
from equitrain.backends.torch_optimizer import create_optimizer_impl
from equitrain.backends.torch_wrappers import AbstractWrapper
from equitrain.finetune.freeze_torch import FreezeFineTuneWrapper


class _ToyMaceLikeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_embedding = torch.nn.Linear(1, 1, bias=False)
        self.interactions = torch.nn.ModuleList(
            [
                torch.nn.Linear(1, 1, bias=False),
                torch.nn.Linear(1, 1, bias=False),
            ]
        )
        self.products = torch.nn.ModuleList(
            [
                torch.nn.Linear(1, 1, bias=False),
                torch.nn.Linear(1, 1, bias=False),
            ]
        )
        self.readouts = torch.nn.ModuleList(
            [
                torch.nn.Linear(1, 1, bias=False),
                torch.nn.Linear(1, 1, bias=False),
            ]
        )

    def forward(self, x):
        x = self.node_embedding(x)
        for interaction, product in zip(self.interactions, self.products, strict=True):
            x = interaction(x)
            x = product(x)
        for readout in self.readouts:
            x = readout(x)
        return x


class _ToyMaceLikeWrapper(AbstractWrapper):
    def __init__(self):
        super().__init__(_ToyMaceLikeModel())

    def forward(self, x):
        return {'energy': self.model(x)}

    @property
    def atomic_numbers(self):
        return None

    @property
    def atomic_energies(self):
        return None

    @property
    def r_max(self):
        return 1.0

    @r_max.setter
    def r_max(self, value):
        del value


def _named_trainable_params(wrapper: FreezeFineTuneWrapper) -> list[str]:
    return [name for name, _param in wrapper.named_fine_tune_parameters()]


def _named_frozen_params(wrapper: FreezeFineTuneWrapper) -> list[str]:
    return [
        name
        for name, param in wrapper.base_wrapper.named_parameters()
        if not param.requires_grad
    ]


def test_freeze_wrapper_defaults_to_all_layers_trainable():
    base_wrapper = _ToyMaceLikeWrapper()
    for param in base_wrapper.parameters():
        param.requires_grad_(False)

    wrapper = FreezeFineTuneWrapper(base_wrapper)

    assert wrapper.freeze_layer_names == (
        'node_embedding',
        'interactions.0',
        'products.0',
        'interactions.1',
        'products.1',
        'readouts',
    )
    assert _named_frozen_params(wrapper) == []


def test_freeze_wrapper_freezes_semantic_layer_range():
    wrapper = FreezeFineTuneWrapper(_ToyMaceLikeWrapper(), freeze_layers='2-')

    assert _named_trainable_params(wrapper) == [
        'model.node_embedding.weight',
        'model.interactions.0.weight',
    ]
    assert _named_frozen_params(wrapper) == [
        'model.interactions.1.weight',
        'model.products.0.weight',
        'model.products.1.weight',
        'model.readouts.0.weight',
        'model.readouts.1.weight',
    ]
    assert wrapper.get_fine_tune_export_config() == {
        'wrapper': 'freeze',
        'freeze_layers': '2-',
    }


def test_args_formatter_includes_freeze_freeze_layers():
    args = type('Args', (), {})()
    args.model = FreezeFineTuneWrapper(_ToyMaceLikeWrapper(), freeze_layers='2-')

    formatted = ArgsFormatter(args).format()

    assert 'fine_tune_export' in formatted
    assert 'wrapper' in formatted
    assert 'freeze' in formatted
    assert 'freeze_layers' in formatted
    assert '2-' in formatted


def test_freeze_wrapper_freezes_from_forward_order_index():
    wrapper = FreezeFineTuneWrapper(_ToyMaceLikeWrapper(), freeze_layers='3-')

    assert _named_trainable_params(wrapper) == [
        'model.node_embedding.weight',
        'model.interactions.0.weight',
        'model.products.0.weight',
    ]
    assert _named_frozen_params(wrapper) == [
        'model.interactions.1.weight',
        'model.products.1.weight',
        'model.readouts.0.weight',
        'model.readouts.1.weight',
    ]


def test_freeze_wrapper_can_update_frozen_layer_selection():
    wrapper = FreezeFineTuneWrapper(_ToyMaceLikeWrapper(), freeze_layers='2-')

    wrapper.freeze_model_layers('1')

    assert _named_frozen_params(wrapper) == ['model.interactions.0.weight']
    assert wrapper.get_fine_tune_export_config() == {
        'wrapper': 'freeze',
        'freeze_layers': '1',
    }

    wrapper.freeze_model_layers()

    assert _named_frozen_params(wrapper) == []
    assert wrapper.get_fine_tune_export_config() == {'wrapper': 'freeze'}


def test_freeze_wrapper_optimizer_sees_only_unfrozen_layers():
    wrapper = FreezeFineTuneWrapper(_ToyMaceLikeWrapper(), freeze_layers='2-')

    optimizer = create_optimizer_impl(
        wrapper,
        optimizer_name='adamw',
        lr=1e-3,
        weight_decay=0.0,
    )

    optimized_ids = {
        id(param) for group in optimizer.param_groups for param in group['params']
    }
    trainable_ids = {id(param) for param in wrapper.fine_tune_parameters()}
    frozen_ids = {
        id(param)
        for _name, param in wrapper.base_wrapper.named_parameters()
        if not param.requires_grad
    }

    assert optimized_ids == trainable_ids
    assert optimized_ids.isdisjoint(frozen_ids)


def test_freeze_wrapper_rejects_out_of_range_layer_selection():
    with pytest.raises(ValueError, match='out of range'):
        FreezeFineTuneWrapper(_ToyMaceLikeWrapper(), freeze_layers='6-')
