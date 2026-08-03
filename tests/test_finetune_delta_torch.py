from __future__ import annotations

import pytest
import torch

from equitrain.backends.torch_optimizer import create_optimizer_impl
from equitrain.backends.torch_wrappers import AbstractWrapper
from equitrain.finetune._layer_selection import infer_semantic_layer_names
from equitrain.finetune.delta_torch import DeltaFineTuneWrapper


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


class _ToyMaceLikeModelWithAuxBlockList(_ToyMaceLikeModel):
    def __init__(self):
        super().__init__()
        self.lr_source_maps = torch.nn.ModuleList(
            [
                torch.nn.Linear(1, 1, bias=False),
                torch.nn.Linear(1, 1, bias=False),
            ]
        )
        self.output_heads = torch.nn.ModuleList([torch.nn.Linear(1, 1, bias=False)])

    def forward(self, x):
        x = self.node_embedding(x)
        for interaction, product, lr_source_map in zip(
            self.interactions,
            self.products,
            self.lr_source_maps,
            strict=True,
        ):
            x = interaction(x)
            x = product(x)
            x = lr_source_map(x)
        for readout in self.readouts:
            x = readout(x)
        for output_head in self.output_heads:
            x = output_head(x)
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


class _ToyMaceLikeWrapperWithAuxBlockList(_ToyMaceLikeWrapper):
    def __init__(self):
        AbstractWrapper.__init__(self, _ToyMaceLikeModelWithAuxBlockList())


def _named_trainable_deltas(wrapper: DeltaFineTuneWrapper) -> list[str]:
    return [
        name for name, delta in wrapper.named_delta_parameters() if delta.requires_grad
    ]


def _named_frozen_deltas(wrapper: DeltaFineTuneWrapper) -> list[str]:
    return [
        name
        for name, delta in wrapper.named_delta_parameters()
        if not delta.requires_grad
    ]


def test_semantic_layer_order_handles_variable_mace_depth():
    parameter_names = ['model.node_embedding.weight']
    parameter_names.extend(f'model.interactions.{index}.weight' for index in range(4))
    parameter_names.extend(f'model.products.{index}.weight' for index in range(4))
    parameter_names.extend(
        [
            'model.readouts.0.weight',
            'model.readouts.1.weight',
        ]
    )

    assert infer_semantic_layer_names(parameter_names) == (
        'node_embedding',
        'interactions.0',
        'products.0',
        'interactions.1',
        'products.1',
        'interactions.2',
        'products.2',
        'interactions.3',
        'products.3',
        'readouts',
    )


def test_delta_wrapper_orders_auxiliary_block_module_lists_with_backbone():
    wrapper = DeltaFineTuneWrapper(_ToyMaceLikeWrapperWithAuxBlockList())

    assert wrapper.delta_layer_names == (
        'node_embedding',
        'interactions.0',
        'products.0',
        'lr_source_maps.0',
        'interactions.1',
        'products.1',
        'lr_source_maps.1',
        'readouts',
        'output_heads.0',
    )


def test_delta_wrapper_defaults_to_all_layers_trainable():
    wrapper = DeltaFineTuneWrapper(_ToyMaceLikeWrapper())

    assert wrapper.delta_layer_names == (
        'node_embedding',
        'interactions.0',
        'products.0',
        'interactions.1',
        'products.1',
        'readouts',
    )
    assert _named_frozen_deltas(wrapper) == []


def test_delta_wrapper_freezes_semantic_layer_range():
    wrapper = DeltaFineTuneWrapper(_ToyMaceLikeWrapper(), freeze_layers='2-')

    assert _named_trainable_deltas(wrapper) == [
        'model.node_embedding.weight',
        'model.interactions.0.weight',
    ]
    assert _named_frozen_deltas(wrapper) == [
        'model.interactions.1.weight',
        'model.products.0.weight',
        'model.products.1.weight',
        'model.readouts.0.weight',
        'model.readouts.1.weight',
    ]
    assert wrapper.get_fine_tune_export_config() == {
        'wrapper': 'delta',
        'freeze_layers': '2-',
    }


def test_delta_wrapper_freezes_from_forward_order_index():
    wrapper = DeltaFineTuneWrapper(_ToyMaceLikeWrapper(), freeze_layers='3-')

    assert _named_trainable_deltas(wrapper) == [
        'model.node_embedding.weight',
        'model.interactions.0.weight',
        'model.products.0.weight',
    ]
    assert _named_frozen_deltas(wrapper) == [
        'model.interactions.1.weight',
        'model.products.1.weight',
        'model.readouts.0.weight',
        'model.readouts.1.weight',
    ]


def test_delta_wrapper_freezes_comma_separated_layers():
    wrapper = DeltaFineTuneWrapper(_ToyMaceLikeWrapper(), freeze_layers='1,3-4')

    assert _named_frozen_deltas(wrapper) == [
        'model.interactions.0.weight',
        'model.interactions.1.weight',
        'model.products.1.weight',
    ]


def test_delta_wrapper_can_update_frozen_layer_selection():
    wrapper = DeltaFineTuneWrapper(_ToyMaceLikeWrapper())

    wrapper.freeze_delta_layers('1')

    assert _named_frozen_deltas(wrapper) == ['model.interactions.0.weight']
    assert wrapper.get_fine_tune_export_config() == {
        'wrapper': 'delta',
        'freeze_layers': '1',
    }

    wrapper.freeze_delta_layers()

    assert _named_frozen_deltas(wrapper) == []
    assert wrapper.get_fine_tune_export_config() == {'wrapper': 'delta'}


def test_delta_wrapper_optimizer_sees_only_unfrozen_layers():
    wrapper = DeltaFineTuneWrapper(_ToyMaceLikeWrapper(), freeze_layers='2-')

    optimizer = create_optimizer_impl(
        wrapper,
        optimizer_name='adamw',
        lr=1e-3,
        weight_decay=0.0,
    )

    optimized_ids = {
        id(param) for group in optimizer.param_groups for param in group['params']
    }
    trainable_ids = {
        id(delta)
        for _name, delta in wrapper.named_delta_parameters()
        if delta.requires_grad
    }
    frozen_ids = {
        id(delta)
        for _name, delta in wrapper.named_delta_parameters()
        if not delta.requires_grad
    }

    assert optimized_ids == trainable_ids
    assert optimized_ids.isdisjoint(frozen_ids)


def test_delta_wrapper_rejects_out_of_range_layer_selection():
    with pytest.raises(ValueError, match='out of range'):
        DeltaFineTuneWrapper(_ToyMaceLikeWrapper(), freeze_layers='6-')
