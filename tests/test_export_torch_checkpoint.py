from __future__ import annotations

import json

import pytest
import torch

from equitrain import get_args_parser_export
from equitrain.backends.torch_wrappers.base import AbstractWrapper
from equitrain.finetune.delta_torch import DeltaFineTuneWrapper
from equitrain.finetune.lora_torch import LoRAFineTuneWrapper
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


def _build_wrapper(weight: float) -> _WrappedLinear:
    model = _WrappedLinear(weight)
    with torch.no_grad():
        model.model.weight.fill_(weight)
    return model


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
