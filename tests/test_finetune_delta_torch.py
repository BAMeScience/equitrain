from __future__ import annotations

import pytest
import torch

from equitrain.argparser import ArgsFilterSimple, ArgsFormatter
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


class _Parameterless(torch.nn.Module):
    def forward(self, x):
        return x


class _SingleParameter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.weight


class _SevenParameterBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.ones(1)) for _ in range(7)]
        )

    def forward(self, x):
        for weight in self.weights:
            x = x * weight
        return x


class _Equitrain010MaceLikeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_embedding = _SingleParameter()
        self.radial_embedding = _Parameterless()
        self.spherical_harmonics = _Parameterless()
        self.atomic_energies_fn = _Parameterless()
        self.interactions = torch.nn.ModuleList(
            [_SevenParameterBlock(), _SevenParameterBlock()]
        )
        self.products = torch.nn.ModuleList([_SingleParameter(), _SingleParameter()])
        self.readouts = torch.nn.ModuleList([_SingleParameter(), _SingleParameter()])
        self.scale_shift = _Parameterless()

    def forward(self, x):
        x = self.node_embedding(x)
        for interaction, product in zip(self.interactions, self.products, strict=True):
            x = interaction(x)
            x = product(x)
        for readout in self.readouts:
            x = readout(x)
        return self.scale_shift(x)


class _ToyMaceLikeWrapper(AbstractWrapper):
    def __init__(self, model=None):
        super().__init__(_ToyMaceLikeModel() if model is None else model)

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


# Equitrain 0.1.0's delta wrapper exposed len(model._modules) positional
# deltas to the optimizer because named_parameters() zipped top-level module
# names with the full delta list. These constants record that legacy contract
# for _Equitrain010MaceLikeModel using current wrapper parameter names.
_EQUITRAIN_0_1_0_TOP_LEVEL_MODULE_NAMES = (
    'node_embedding',
    'radial_embedding',
    'spherical_harmonics',
    'atomic_energies_fn',
    'interactions',
    'products',
    'readouts',
    'scale_shift',
)
_EQUITRAIN_0_1_0_DELTA_VALUES = (
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
)
_EQUITRAIN_0_1_0_TRAINABLE_BASE_PARAMETER_NAMES = (
    'model.node_embedding.weight',
    'model.interactions.0.weights.0',
    'model.interactions.0.weights.1',
    'model.interactions.0.weights.2',
    'model.interactions.0.weights.3',
    'model.interactions.0.weights.4',
    'model.interactions.0.weights.5',
    'model.interactions.0.weights.6',
)
_EQUITRAIN_0_1_0_INPUT = (
    (1.5,),
    (2.0,),
)
_EQUITRAIN_0_1_0_TARGET_ENERGY = (
    (2.1290507316589355,),
    (2.8387346267700195,),
)
# Generated with one Equitrain 0.1.0 AdamW optimizer step from the delta
# values above, using MSE loss against this fixed target.
_EQUITRAIN_0_1_0_ONE_STEP_TARGET = (
    (1.0,),
    (1.25,),
)
_EQUITRAIN_0_1_0_ONE_STEP_LOSS_BEFORE = 1.8994166851043701
_EQUITRAIN_0_1_0_ONE_STEP_UPDATED_DELTA_VALUES = (
    -0.000009999610483646393,
    0.00998000055551529,
    0.01996999979019165,
    0.02996000088751316,
    0.03994999825954437,
    0.049939997494220734,
    0.05993000417947769,
    0.06992000341415405,
)
_EQUITRAIN_0_1_0_ONE_STEP_PREDICTION_AFTER = (
    (1.9706650972366333,),
    (2.627553701400757,),
)
_EQUITRAIN_0_1_0_ONE_STEP_LOSS_AFTER = 1.4199224710464478


def _set_equitrain_0_1_0_deltas(wrapper: DeltaFineTuneWrapper) -> None:
    with torch.no_grad():
        for index, (_name, delta) in enumerate(wrapper.named_delta_parameters()):
            if index < len(_EQUITRAIN_0_1_0_DELTA_VALUES):
                delta.fill_(_EQUITRAIN_0_1_0_DELTA_VALUES[index])
            else:
                delta.zero_()


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


def test_delta_wrapper_legacy_selection_matches_equitrain_0_1_0():
    wrapper = DeltaFineTuneWrapper(
        _ToyMaceLikeWrapper(_Equitrain010MaceLikeModel()), freeze_layers='2-'
    )

    assert tuple(wrapper.model._modules) == _EQUITRAIN_0_1_0_TOP_LEVEL_MODULE_NAMES
    assert wrapper.delta_layer_names == (
        'node_embedding',
        'interactions.0',
        'products.0',
        'interactions.1',
        'products.1',
        'readouts',
    )
    assert _named_trainable_deltas(wrapper) == list(
        _EQUITRAIN_0_1_0_TRAINABLE_BASE_PARAMETER_NAMES
    )


def test_delta_wrapper_legacy_predictions_match_equitrain_0_1_0():
    wrapper = DeltaFineTuneWrapper(
        _ToyMaceLikeWrapper(_Equitrain010MaceLikeModel()), freeze_layers='2-'
    )
    _set_equitrain_0_1_0_deltas(wrapper)

    x = torch.tensor(_EQUITRAIN_0_1_0_INPUT, dtype=torch.float32)
    target_energy = torch.tensor(_EQUITRAIN_0_1_0_TARGET_ENERGY, dtype=torch.float32)

    torch.testing.assert_close(wrapper(x)['energy'], target_energy)


def test_delta_wrapper_one_optimizer_step_matches_equitrain_0_1_0():
    wrapper = DeltaFineTuneWrapper(
        _ToyMaceLikeWrapper(_Equitrain010MaceLikeModel()), freeze_layers='2-'
    )
    _set_equitrain_0_1_0_deltas(wrapper)
    optimizer = create_optimizer_impl(
        wrapper,
        optimizer_name='adamw',
        lr=0.01,
        weight_decay=0.1,
        alpha=0.99,
        momentum=0.0,
    )

    assert [len(group['params']) for group in optimizer.param_groups] == [
        0,
        len(_EQUITRAIN_0_1_0_DELTA_VALUES),
    ]

    x = torch.tensor(_EQUITRAIN_0_1_0_INPUT, dtype=torch.float32)
    target = torch.tensor(_EQUITRAIN_0_1_0_ONE_STEP_TARGET, dtype=torch.float32)

    optimizer.zero_grad()
    prediction_before = wrapper(x)['energy']
    loss_before = torch.nn.functional.mse_loss(prediction_before, target)
    torch.testing.assert_close(
        loss_before.detach(),
        torch.tensor(_EQUITRAIN_0_1_0_ONE_STEP_LOSS_BEFORE),
    )

    loss_before.backward()
    optimizer.step()

    delta_entries = list(wrapper.named_delta_parameters())
    visible_delta_values = torch.stack(
        [
            delta.detach().reshape(())
            for _name, delta in delta_entries[: len(_EQUITRAIN_0_1_0_DELTA_VALUES)]
        ]
    )
    torch.testing.assert_close(
        visible_delta_values,
        torch.tensor(_EQUITRAIN_0_1_0_ONE_STEP_UPDATED_DELTA_VALUES),
    )
    assert all(
        torch.count_nonzero(delta.detach()).item() == 0
        for _name, delta in delta_entries[len(_EQUITRAIN_0_1_0_DELTA_VALUES) :]
    )

    prediction_after = wrapper(x)['energy']
    torch.testing.assert_close(
        prediction_after.detach(),
        torch.tensor(_EQUITRAIN_0_1_0_ONE_STEP_PREDICTION_AFTER),
    )
    torch.testing.assert_close(
        torch.nn.functional.mse_loss(prediction_after, target).detach(),
        torch.tensor(_EQUITRAIN_0_1_0_ONE_STEP_LOSS_AFTER),
    )


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


def test_args_formatter_includes_delta_freeze_layers():
    args = type('Args', (), {})()
    args.model = DeltaFineTuneWrapper(_ToyMaceLikeWrapper(), freeze_layers='2-')

    formatted = ArgsFormatter(args).format()

    assert 'fine_tune_export' in formatted
    assert 'wrapper' in formatted
    assert 'delta' in formatted
    assert 'freeze_layers' in formatted
    assert '2-' in formatted


def test_args_filter_simple_includes_delta_freeze_layers():
    args = type('Args', (), {})()
    args.model = DeltaFineTuneWrapper(_ToyMaceLikeWrapper(), freeze_layers='2-')
    args.lr = 1e-3

    filtered = ArgsFilterSimple().filter(args)

    assert filtered['fine_tune_export'] == {
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
