from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from equitrain import get_args_parser_export
from equitrain.backends.torch_checkpoint import save_checkpoint
from equitrain.backends.torch_wrappers.base import AbstractWrapper
from equitrain.finetune._torch_common import uniquify_empty_tensor_storage
from equitrain.finetune.delta_torch import DeltaFineTuneWrapper
from equitrain.finetune.freeze_torch import FreezeFineTuneWrapper
from equitrain.finetune.lora_torch import LoRAFineTuneWrapper
from equitrain.logger import FileLogger
from equitrain.scripts.equitrain_export import export


class _WrappedLinear(AbstractWrapper):
    def __init__(self, weight: float) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(weight)
        super().__init__(model)
        self._r_max = 1.0

    def forward(self, *args, **kwargs):
        raise NotImplementedError('Forward is not used by the export checkpoint tests.')

    @property
    def atomic_numbers(self):
        return torch.tensor([1])

    @property
    def atomic_energies(self):
        return None

    @property
    def r_max(self):
        return self._r_max

    @r_max.setter
    def r_max(self, value):
        self._r_max = value


class _ViewTensorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = torch.nn.Parameter(torch.arange(4.0))
        self.register_parameter('base', base)
        self.register_buffer('view', base[:2])


class _WrappedViewTensorModel(AbstractWrapper):
    def __init__(self) -> None:
        super().__init__(_ViewTensorModel())
        self._r_max = 1.0

    def forward(self, *args, **kwargs):
        raise NotImplementedError('Forward is not used by the export checkpoint tests.')

    @property
    def atomic_numbers(self):
        return torch.tensor([1])

    @property
    def atomic_energies(self):
        return None

    @property
    def r_max(self):
        return self._r_max

    @r_max.setter
    def r_max(self, value):
        self._r_max = value


class _EmptyTensorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Parameter(torch.empty(0))
        self.second = torch.nn.Parameter(torch.empty(0))


class _WrappedEmptyTensorModel(AbstractWrapper):
    def __init__(self) -> None:
        super().__init__(_EmptyTensorModel())
        self._r_max = 1.0

    def forward(self, *args, **kwargs):
        raise NotImplementedError('Forward is not used by the export checkpoint tests.')

    @property
    def atomic_numbers(self):
        return torch.tensor([1])

    @property
    def atomic_energies(self):
        return None

    @property
    def r_max(self):
        return self._r_max

    @r_max.setter
    def r_max(self, value):
        self._r_max = value


class _FakeAccelerator:
    def __init__(self, model):
        self.model = model

    def save_state(self, output_dir):
        torch.save(self.model.state_dict(), output_dir / 'pytorch_model.bin')

    def unwrap_model(self, model):
        return model


def _build_wrapper(weight: float) -> _WrappedLinear:
    model = _WrappedLinear(weight)
    with torch.no_grad():
        model.model.weight.fill_(weight)
    return model


def _shared_tensor_groups(state_dict):
    groups = {}
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            groups.setdefault(tensor.untyped_storage().data_ptr(), []).append(name)
    return [names for names in groups.values() if len(names) > 1]


def test_delta_state_dict_registers_base_model_once():
    model = DeltaFineTuneWrapper(_build_wrapper(1.0))
    state_dict = model.state_dict()

    assert 'model.weight' not in state_dict
    assert 'base_wrapper.model.weight' in state_dict
    assert '_delta_params.model__DOT__weight' in state_dict
    assert _shared_tensor_groups(state_dict) == []


def test_lora_state_dict_registers_base_model_once():
    model = LoRAFineTuneWrapper(_build_wrapper(1.0), rank_reduction=75, alpha=16)
    state_dict = model.state_dict()

    assert 'model.weight' not in state_dict
    assert 'base_wrapper.model.weight' in state_dict
    assert '_lora_a_params.model__DOT__weight__LORA_A__' in state_dict
    assert '_lora_b_params.model__DOT__weight__LORA_B__' in state_dict
    assert _shared_tensor_groups(state_dict) == []


def test_freeze_state_dict_registers_base_model_once():
    model = FreezeFineTuneWrapper(_build_wrapper(1.0))
    state_dict = model.state_dict()

    assert 'model.weight' not in state_dict
    assert 'base_wrapper.model.weight' in state_dict
    assert _shared_tensor_groups(state_dict) == []


def test_delta_checkpoint_state_dict_uniquifies_empty_tensor_storage():
    model = DeltaFineTuneWrapper(_WrappedEmptyTensorModel())
    state_dict = model.state_dict()
    uniquify_empty_tensor_storage(state_dict)

    assert {
        'base_wrapper.model.first',
        'base_wrapper.model.second',
        '_delta_params.model__DOT__first',
        '_delta_params.model__DOT__second',
    }.issubset(state_dict)
    assert _shared_tensor_groups(state_dict) == []


def test_lora_checkpoint_state_dict_uniquifies_empty_tensor_storage():
    model = LoRAFineTuneWrapper(_WrappedEmptyTensorModel(), rank_reduction=75, alpha=16)
    state_dict = model.state_dict()
    uniquify_empty_tensor_storage(state_dict)

    assert {'base_wrapper.model.first', 'base_wrapper.model.second'}.issubset(
        state_dict
    )
    assert _shared_tensor_groups(state_dict) == []


def test_freeze_checkpoint_state_dict_uniquifies_empty_tensor_storage():
    model = FreezeFineTuneWrapper(_WrappedEmptyTensorModel())
    state_dict = model.state_dict()
    uniquify_empty_tensor_storage(state_dict)

    assert {'base_wrapper.model.first', 'base_wrapper.model.second'}.issubset(
        state_dict
    )
    assert _shared_tensor_groups(state_dict) == []


def test_delta_state_dict_clones_non_complete_tensor_views():
    model = DeltaFineTuneWrapper(_WrappedViewTensorModel())
    state_dict = model.state_dict()

    assert 'base_wrapper.model.view' in state_dict
    view = state_dict['base_wrapper.model.view']
    assert view.untyped_storage().nbytes() == view.numel() * view.element_size()
    assert _shared_tensor_groups(state_dict) == []


def test_lora_state_dict_clones_non_complete_tensor_views():
    model = LoRAFineTuneWrapper(_WrappedViewTensorModel(), rank_reduction=75, alpha=16)
    state_dict = model.state_dict()

    assert 'base_wrapper.model.view' in state_dict
    view = state_dict['base_wrapper.model.view']
    assert view.untyped_storage().nbytes() == view.numel() * view.element_size()
    assert _shared_tensor_groups(state_dict) == []


def test_freeze_state_dict_clones_non_complete_tensor_views():
    model = FreezeFineTuneWrapper(_WrappedViewTensorModel())
    state_dict = model.state_dict()

    assert 'base_wrapper.model.view' in state_dict
    view = state_dict['base_wrapper.model.view']
    assert view.untyped_storage().nbytes() == view.numel() * view.element_size()
    assert _shared_tensor_groups(state_dict) == []


def test_delta_loads_legacy_top_level_model_keys():
    model = DeltaFineTuneWrapper(_build_wrapper(1.0))
    legacy_state = {
        'model.weight': torch.full_like(model.model.weight, 4.0),
        '_delta_params.model__DOT__weight': torch.full_like(
            dict(model.named_delta_parameters())['model.weight'], 2.0
        ),
    }

    model.load_state_dict(legacy_state)

    torch.testing.assert_close(
        model.model.weight, torch.full_like(model.model.weight, 4.0)
    )


def test_lora_loads_legacy_top_level_model_keys():
    model = LoRAFineTuneWrapper(_build_wrapper(1.0), rank_reduction=75, alpha=16)
    named_lora = dict(model.named_lora_parameters())
    legacy_state = {
        'model.weight': torch.full_like(model.model.weight, 4.0),
        '_lora_a_params.model__DOT__weight__LORA_A__': torch.full_like(
            named_lora['model.weight.lora_a'], 0.25
        ),
        '_lora_b_params.model__DOT__weight__LORA_B__': torch.full_like(
            named_lora['model.weight.lora_b'], 0.5
        ),
    }

    model.load_state_dict(legacy_state)

    torch.testing.assert_close(
        model.model.weight, torch.full_like(model.model.weight, 4.0)
    )


def test_freeze_loads_legacy_top_level_model_keys():
    model = FreezeFineTuneWrapper(_build_wrapper(1.0))
    legacy_state = {'model.weight': torch.full_like(model.model.weight, 4.0)}

    model.load_state_dict(legacy_state)

    torch.testing.assert_close(
        model.model.weight, torch.full_like(model.model.weight, 4.0)
    )


def _save_export_inputs(tmp_path, *, checkpoint_name: str):
    output_dir = tmp_path / 'training'
    checkpoint_dir = output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    base_model = _build_wrapper(1.0)
    trained_model = _build_wrapper(5.0)

    base_model_path = tmp_path / 'base.model'
    export_path = tmp_path / 'exported.model'

    torch.save(base_model, base_model_path)
    torch.save(trained_model.state_dict(), checkpoint_dir / 'pytorch_model.bin')
    (checkpoint_dir / 'args.json').write_text('{}')

    return (
        base_model,
        trained_model,
        base_model_path,
        export_path,
        output_dir,
        checkpoint_dir,
    )


def test_export_loads_best_checkpoint_directory(tmp_path):
    (
        base_model,
        trained_model,
        base_model_path,
        export_path,
        output_dir,
        _checkpoint_dir,
    ) = _save_export_inputs(tmp_path, checkpoint_name='best_val_epochs@3_e@0.1')

    args = get_args_parser_export().parse_args(
        [
            '--model',
            str(base_model_path),
            '--output-dir',
            str(output_dir),
            '--load-best-checkpoint',
            '--model-export',
            str(export_path),
        ]
    )

    export(args)

    exported = torch.load(export_path, weights_only=False)

    assert torch.equal(exported.weight, trained_model.model.weight)
    assert not torch.equal(exported.weight, base_model.model.weight)


def test_export_loads_explicit_checkpoint_directory(tmp_path):
    (
        base_model,
        trained_model,
        base_model_path,
        export_path,
        _output_dir,
        checkpoint_dir,
    ) = _save_export_inputs(tmp_path, checkpoint_name='best_val_epochs@7_e@0.2')

    args = get_args_parser_export().parse_args(
        [
            '--model',
            str(base_model_path),
            '--load-checkpoint',
            str(checkpoint_dir),
            '--model-export',
            str(export_path),
        ]
    )

    export(args)

    exported = torch.load(export_path, weights_only=False)

    assert torch.equal(exported.weight, trained_model.model.weight)
    assert not torch.equal(exported.weight, base_model.model.weight)


def _save_adapter_export_inputs(tmp_path, adapter, *, include_config: bool = True):
    output_dir = tmp_path / f'{adapter}_training'
    checkpoint_dir = output_dir / 'best_val_epochs@4_e@0.2'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    base_model = _build_wrapper(1.0)
    base_model_path = tmp_path / f'{adapter}_base.model'
    export_path = tmp_path / f'{adapter}_exported.model'
    torch.save(base_model, base_model_path)

    if adapter == 'delta':
        trained_model = DeltaFineTuneWrapper(_build_wrapper(1.0))
        with torch.no_grad():
            for delta in trained_model.delta_parameters():
                delta.fill_(2.0)
    elif adapter == 'lora':
        trained_model = LoRAFineTuneWrapper(
            _build_wrapper(1.0),
            rank_reduction=75,
            alpha=16,
        )
        named_lora = dict(trained_model.named_lora_parameters())
        with torch.no_grad():
            named_lora['model.weight.lora_a'].copy_(torch.tensor([[0.25, 0.5]]))
            named_lora['model.weight.lora_b'].copy_(torch.tensor([[0.5]]))
    elif adapter == 'freeze':
        trained_model = FreezeFineTuneWrapper(_build_wrapper(1.0), freeze_layers='0')
        with torch.no_grad():
            trained_model.model.weight.fill_(4.0)
    else:  # pragma: no cover - defensive guard for test helper use
        raise ValueError(adapter)

    state_dict = trained_model.state_dict()
    torch.save(state_dict, checkpoint_dir / 'pytorch_model.bin')
    args_payload = {}
    if include_config:
        config_fn = getattr(trained_model, 'get_fine_tune_export_config', None)
        if callable(config_fn):
            args_payload['fine_tune_export'] = config_fn()
    (checkpoint_dir / 'args.json').write_text(json.dumps(args_payload))

    return base_model_path, export_path, output_dir, checkpoint_dir


def test_export_auto_loads_delta_checkpoint(tmp_path):
    base_model_path, export_path, output_dir, _checkpoint_dir = (
        _save_adapter_export_inputs(tmp_path, 'delta')
    )

    args = get_args_parser_export().parse_args(
        [
            '--model',
            str(base_model_path),
            '--output-dir',
            str(output_dir),
            '--load-best-checkpoint',
            '--model-export',
            str(export_path),
        ]
    )

    export(args)

    exported = torch.load(export_path, weights_only=False)
    torch.testing.assert_close(exported.weight, torch.full_like(exported.weight, 3.0))


def test_export_auto_loads_lora_checkpoint_with_metadata(tmp_path):
    base_model_path, export_path, output_dir, _checkpoint_dir = (
        _save_adapter_export_inputs(tmp_path, 'lora')
    )

    args = get_args_parser_export().parse_args(
        [
            '--model',
            str(base_model_path),
            '--output-dir',
            str(output_dir),
            '--load-best-checkpoint',
            '--model-export',
            str(export_path),
        ]
    )

    export(args)

    exported = torch.load(export_path, weights_only=False)
    expected = torch.tensor([[3.0, 5.0]], dtype=exported.weight.dtype)
    torch.testing.assert_close(exported.weight, expected)


def test_export_auto_loads_freeze_checkpoint_with_metadata(tmp_path):
    base_model_path, export_path, output_dir, _checkpoint_dir = (
        _save_adapter_export_inputs(tmp_path, 'freeze')
    )

    args = get_args_parser_export().parse_args(
        [
            '--model',
            str(base_model_path),
            '--output-dir',
            str(output_dir),
            '--load-best-checkpoint',
            '--model-export',
            str(export_path),
        ]
    )

    export(args)

    exported = torch.load(export_path, weights_only=False)
    torch.testing.assert_close(exported.weight, torch.full_like(exported.weight, 4.0))


def test_export_auto_detects_freeze_checkpoint_without_metadata(tmp_path):
    base_model_path, export_path, output_dir, _checkpoint_dir = (
        _save_adapter_export_inputs(tmp_path, 'freeze', include_config=False)
    )

    args = get_args_parser_export().parse_args(
        [
            '--model',
            str(base_model_path),
            '--output-dir',
            str(output_dir),
            '--load-best-checkpoint',
            '--model-export',
            str(export_path),
        ]
    )

    export(args)

    exported = torch.load(export_path, weights_only=False)
    torch.testing.assert_close(exported.weight, torch.full_like(exported.weight, 4.0))


@pytest.mark.parametrize(
    ('adapter', 'expected'),
    [
        ('delta', {'wrapper': 'delta'}),
        ('freeze', {'wrapper': 'freeze'}),
        (
            'lora',
            {
                'wrapper': 'lora',
                'rank_fraction': None,
                'rank_reduction': 75,
                'min_rank': 1,
                'alpha': 16,
            },
        ),
    ],
)
def test_save_checkpoint_writes_fine_tune_export_metadata(tmp_path, adapter, expected):
    if adapter == 'delta':
        model = DeltaFineTuneWrapper(_build_wrapper(1.0))
    elif adapter == 'freeze':
        model = FreezeFineTuneWrapper(_build_wrapper(1.0))
    elif adapter == 'lora':
        model = LoRAFineTuneWrapper(
            _build_wrapper(1.0),
            rank_reduction=75,
            alpha=16,
        )
    else:  # pragma: no cover - defensive guard for parametrized values
        raise ValueError(adapter)

    args = SimpleNamespace(output_dir=str(tmp_path), verbose=0)
    valid_loss = {'total': SimpleNamespace(avg=0.2)}
    logger = FileLogger(enable_logging=False, stream=False)

    checkpoint_dir = save_checkpoint(
        args,
        epoch=4,
        valid_loss=valid_loss,
        model_ema=None,
        accelerator=_FakeAccelerator(model),
        logger=logger,
        model=model,
    )

    args_payload = json.loads((checkpoint_dir / 'args.json').read_text())
    assert args_payload['fine_tune_export'] == expected


def test_save_checkpoint_writes_delta_freeze_layers_metadata(tmp_path):
    model = DeltaFineTuneWrapper(_build_wrapper(1.0), freeze_layers='0')
    args = SimpleNamespace(output_dir=str(tmp_path), verbose=0)
    valid_loss = {'total': SimpleNamespace(avg=0.2)}
    logger = FileLogger(enable_logging=False, stream=False)

    checkpoint_dir = save_checkpoint(
        args,
        epoch=4,
        valid_loss=valid_loss,
        model_ema=None,
        accelerator=_FakeAccelerator(model),
        logger=logger,
        model=model,
    )

    args_payload = json.loads((checkpoint_dir / 'args.json').read_text())
    assert args_payload['fine_tune_export'] == {
        'wrapper': 'delta',
        'freeze_layers': '0',
    }


def test_save_checkpoint_writes_freeze_layers_metadata(tmp_path):
    model = FreezeFineTuneWrapper(_build_wrapper(1.0), freeze_layers='0')
    args = SimpleNamespace(output_dir=str(tmp_path), verbose=0)
    valid_loss = {'total': SimpleNamespace(avg=0.2)}
    logger = FileLogger(enable_logging=False, stream=False)

    checkpoint_dir = save_checkpoint(
        args,
        epoch=4,
        valid_loss=valid_loss,
        model_ema=None,
        accelerator=_FakeAccelerator(model),
        logger=logger,
        model=model,
    )

    args_payload = json.loads((checkpoint_dir / 'args.json').read_text())
    assert args_payload['fine_tune_export'] == {
        'wrapper': 'freeze',
        'freeze_layers': '0',
    }


def test_export_rejects_adapter_checkpoint_without_metadata(tmp_path):
    base_model_path, export_path, output_dir, _checkpoint_dir = (
        _save_adapter_export_inputs(tmp_path, 'lora', include_config=False)
    )

    args = get_args_parser_export().parse_args(
        [
            '--model',
            str(base_model_path),
            '--output-dir',
            str(output_dir),
            '--load-best-checkpoint',
            '--model-export',
            str(export_path),
        ]
    )

    with pytest.raises(ValueError, match='fine_tune_export metadata'):
        export(args)
