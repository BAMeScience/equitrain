from __future__ import annotations

import numpy as np
import torch

from equitrain.backends.torch_wrappers import AbstractWrapper
from equitrain.finetune.lora_torch import LoRAFineTuneWrapper


class _ToyTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3, bias=True)
        self.conv = torch.nn.Conv1d(1, 2, kernel_size=3, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10.0
            )
            self.linear.bias.copy_(torch.arange(3, dtype=torch.float32) / 10.0)
            self.conv.weight.copy_(
                torch.arange(6, dtype=torch.float32).reshape(2, 1, 3) / 10.0
            )

    def forward(self, linear_x, conv_x):
        linear_term = self.linear(linear_x).sum(dim=-1)
        conv_term = self.conv(conv_x).sum(dim=(1, 2))
        return linear_term + conv_term


class _ToyTorchWrapper(AbstractWrapper):
    def __init__(self):
        super().__init__(_ToyTorchModel())

    def forward(self, linear_x, conv_x):
        return {'energy': self.model(linear_x, conv_x)}

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


def test_torch_lora_wrapper_uses_percentage_rank_reduction():
    base_wrapper = _ToyTorchWrapper()
    lora_wrapper = LoRAFineTuneWrapper(base_wrapper, rank_reduction=50)

    named_lora = dict(lora_wrapper.named_lora_parameters())
    assert set(named_lora) == {
        'model.linear.weight.lora_a',
        'model.linear.weight.lora_b',
        'model.conv.weight.lora_a',
        'model.conv.weight.lora_b',
    }
    assert 'model.linear.bias' not in lora_wrapper.lora_specs

    linear_spec = lora_wrapper.lora_specs['model.linear.weight']
    conv_spec = lora_wrapper.lora_specs['model.conv.weight']
    assert linear_spec.rank == 2
    assert conv_spec.rank == 1
    assert tuple(named_lora['model.linear.weight.lora_a'].shape) == (2, 4)
    assert tuple(named_lora['model.linear.weight.lora_b'].shape) == (3, 2)
    assert tuple(named_lora['model.conv.weight.lora_a'].shape) == (1, 3)
    assert tuple(named_lora['model.conv.weight.lora_b'].shape) == (2, 1)


def test_torch_lora_wrapper_preserves_base_output_until_updated():
    base_wrapper = _ToyTorchWrapper()
    lora_wrapper = LoRAFineTuneWrapper(base_wrapper, rank_reduction=50)

    linear_x = torch.tensor([[1.0, -1.0, 0.5, 2.0]], dtype=torch.float32)
    conv_x = torch.tensor([[[1.0, -2.0, 0.5, 3.0]]], dtype=torch.float32)

    base_energy = base_wrapper(linear_x, conv_x)['energy'].detach().cpu().numpy()
    lora_energy = lora_wrapper(linear_x, conv_x)['energy'].detach().cpu().numpy()
    np.testing.assert_allclose(lora_energy, base_energy)

    linear_weight_before = base_wrapper.model.linear.weight.detach().clone()
    with torch.no_grad():
        dict(lora_wrapper.named_lora_parameters())['model.linear.weight.lora_b'].fill_(
            0.25
        )

    updated_energy = lora_wrapper(linear_x, conv_x)['energy'].detach().cpu().numpy()
    assert not np.allclose(updated_energy, base_energy)
    torch.testing.assert_close(base_wrapper.model.linear.weight, linear_weight_before)
