from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('jax', reason='JAX runtime is required for JAX LoRA tests.')
pytest.importorskip('flax', reason='Flax is required for JAX LoRA tests.')
pytest.importorskip('mace_jax', reason='mace_jax is required for JAX LoRA tests.')

import jax.numpy as jnp  # noqa: E402
from flax import core as flax_core  # noqa: E402
from flax import nnx, traverse_util  # noqa: E402

from equitrain.finetune.lora_jax import LoRAFineTuneModule


class _LinearModule(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 10.0)
        self.bias = nnx.Param(jnp.arange(3, dtype=jnp.float32) / 10.0)

    def __call__(self, x):
        return jnp.matmul(x, self.weight.value.T) + self.bias.value


class _TensorModule(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(
            jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3) / 10.0
        )

    def __call__(self, x):
        return jnp.einsum('bij,oij->bo', x, self.weight.value)


class _ToyJaxModule(nnx.Module):
    def __init__(self):
        self.linear = _LinearModule()
        self.tensor = _TensorModule()

    def __call__(self, linear_x, tensor_x):
        linear_term = jnp.sum(self.linear(linear_x), axis=-1)
        tensor_term = jnp.sum(self.tensor(tensor_x), axis=-1)
        return {'energy': linear_term + tensor_term}


def test_jax_lora_wrapper_uses_percentage_rank_reduction():
    module = _ToyJaxModule()
    lora_module = LoRAFineTuneModule(module, rank_reduction=50)
    variables = lora_module.init()
    flat_lora = traverse_util.flatten_dict(
        flax_core.unfreeze(variables['params']['lora'])
    )

    assert ('linear', 'weight', 'a') in flat_lora
    assert ('linear', 'weight', 'b') in flat_lora
    assert ('tensor', 'weight', 'a') in flat_lora
    assert ('tensor', 'weight', 'b') in flat_lora
    assert ('linear', 'bias', 'a') not in flat_lora

    linear_spec = lora_module.lora_specs[('linear', 'weight')]
    tensor_spec = lora_module.lora_specs[('tensor', 'weight')]
    assert linear_spec.rank == 2
    assert tensor_spec.rank == 1
    assert flat_lora[('linear', 'weight', 'a')].shape == (2, 4)
    assert flat_lora[('linear', 'weight', 'b')].shape == (3, 2)
    assert flat_lora[('tensor', 'weight', 'a')].shape == (1, 6)
    assert flat_lora[('tensor', 'weight', 'b')].shape == (2, 1)


def test_jax_lora_wrapper_preserves_base_output_until_updated():
    module = _ToyJaxModule()
    lora_module = LoRAFineTuneModule(module, rank_reduction=50)
    variables = lora_module.init()

    linear_x = jnp.array([[1.0, -1.0, 0.5, 2.0]], dtype=jnp.float32)
    tensor_x = jnp.array(
        [[[1.0, -2.0, 0.5], [0.5, 1.5, -1.0]]],
        dtype=jnp.float32,
    )

    base_energy = np.asarray(module(linear_x, tensor_x)['energy'])
    lora_energy = np.asarray(lora_module.apply(variables, linear_x, tensor_x)['energy'])
    np.testing.assert_allclose(lora_energy, base_energy)

    updated = flax_core.unfreeze(variables)
    updated['params']['lora']['linear']['weight']['b'] = jnp.full_like(
        updated['params']['lora']['linear']['weight']['b'],
        0.25,
    )
    updated = flax_core.freeze(updated)

    updated_energy = np.asarray(
        lora_module.apply(updated, linear_x, tensor_x)['energy']
    )
    assert not np.allclose(updated_energy, base_energy)
    np.testing.assert_allclose(
        np.asarray(module(linear_x, tensor_x)['energy']),
        base_energy,
    )
