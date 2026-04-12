from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import core as flax_core
from flax import nnx, traverse_util
from mace_jax.nnx_utils import state_to_pure_dict

from equitrain.finetune._lora_common import (
    effective_matrix_shape,
    resolve_rank,
    resolve_retained_fraction,
)


def _as_mutable_tree(tree):
    return flax_core.unfreeze(tree)


def _is_lora_param_path(path: tuple[object, ...], value) -> bool:
    arr = jnp.asarray(value)
    return bool(path) and str(path[-1]) == 'weight' and arr.ndim >= 2


def _path_seed(path: tuple[object, ...]) -> int:
    raw = '/'.join(map(str, path)).encode()
    return int(zlib.crc32(raw) & 0xFFFFFFFF)


def _init_lora_a(
    path: tuple[object, ...],
    *,
    rank: int,
    in_dim: int,
    dtype,
):
    key = jax.random.PRNGKey(_path_seed(path))
    scale = jnp.asarray(1.0 / math.sqrt(in_dim), dtype=dtype)
    return jax.random.normal(key, (rank, in_dim), dtype=dtype) * scale


@dataclass(frozen=True)
class LoRASpec:
    path: tuple[object, ...]
    shape: tuple[int, ...]
    out_dim: int
    in_dim: int
    rank: int
    scale: float


def build_lora_spec_map_from_param_tree(
    param_tree,
    *,
    rank_fraction: float | int | None = None,
    rank_reduction: float | int | None = None,
    min_rank: int = 1,
    alpha: float | None = None,
) -> dict[tuple[object, ...], LoRASpec]:
    retained_fraction = resolve_retained_fraction(
        rank_fraction=rank_fraction,
        rank_reduction=rank_reduction,
    )
    flat_params = traverse_util.flatten_dict(_as_mutable_tree(param_tree))
    lora_specs: dict[tuple[object, ...], LoRASpec] = {}
    for path, value in flat_params.items():
        if not _is_lora_param_path(path, value):
            continue
        arr = jnp.asarray(value)
        out_dim, in_dim = effective_matrix_shape(tuple(arr.shape))
        rank = resolve_rank(
            min(out_dim, in_dim),
            retained_fraction=retained_fraction,
            min_rank=int(min_rank),
        )
        scale = 1.0 if alpha is None else float(alpha) / float(rank)
        lora_specs[path] = LoRASpec(
            path=path,
            shape=tuple(arr.shape),
            out_dim=out_dim,
            in_dim=in_dim,
            rank=rank,
            scale=scale,
        )
    return lora_specs


def build_lora_template_from_param_tree(
    param_tree,
    lora_specs: dict[tuple[object, ...], LoRASpec],
) -> dict:
    flat_params = traverse_util.flatten_dict(_as_mutable_tree(param_tree))
    lora_flat = {}
    for path, spec in lora_specs.items():
        dtype = jnp.asarray(flat_params[path]).dtype
        lora_flat[path + ('a',)] = _init_lora_a(
            path,
            rank=spec.rank,
            in_dim=spec.in_dim,
            dtype=dtype,
        )
        lora_flat[path + ('b',)] = jnp.zeros(
            (spec.out_dim, spec.rank),
            dtype=dtype,
        )
    return traverse_util.unflatten_dict(lora_flat)


def build_lora_spec_map(
    module,
    *,
    rank_fraction: float | int | None = None,
    rank_reduction: float | int | None = None,
    min_rank: int = 1,
    alpha: float | None = None,
) -> dict[tuple[object, ...], LoRASpec]:
    param_tree = state_to_pure_dict(nnx.state(module, nnx.Param))
    return build_lora_spec_map_from_param_tree(
        param_tree,
        rank_fraction=rank_fraction,
        rank_reduction=rank_reduction,
        min_rank=min_rank,
        alpha=alpha,
    )


def build_lora_template(
    module,
    *,
    rank_fraction: float | int | None = None,
    rank_reduction: float | int | None = None,
    min_rank: int = 1,
    alpha: float | None = None,
) -> dict:
    param_tree = state_to_pure_dict(nnx.state(module, nnx.Param))
    lora_specs = build_lora_spec_map_from_param_tree(
        param_tree,
        rank_fraction=rank_fraction,
        rank_reduction=rank_reduction,
        min_rank=min_rank,
        alpha=alpha,
    )
    return build_lora_template_from_param_tree(param_tree, lora_specs)


def merge_lora_params(base_params, lora_params, lora_specs) -> dict:
    flat_base = traverse_util.flatten_dict(_as_mutable_tree(base_params))
    flat_lora = traverse_util.flatten_dict(_as_mutable_tree(lora_params))

    for path, spec in lora_specs.items():
        lora_a = jnp.asarray(flat_lora[path + ('a',)])
        lora_b = jnp.asarray(flat_lora[path + ('b',)])
        delta = jnp.matmul(lora_b, lora_a).reshape(spec.shape)
        flat_base[path] = (
            jax.lax.stop_gradient(jnp.asarray(flat_base[path])) + spec.scale * delta
        )

    return traverse_util.unflatten_dict(flat_base)


def ensure_lora_params(variables, lora_template) -> flax_core.FrozenDict:
    unfrozen = _as_mutable_tree(variables)

    if 'base_params' in unfrozen and unfrozen.get('params', {}).get('lora') is not None:
        return flax_core.freeze(unfrozen)

    return flax_core.freeze(
        {
            'base_params': unfrozen,
            'params': {'lora': _as_mutable_tree(lora_template)},
        }
    )


class LoRAFineTuneModule:
    """
    Wrap an NNX module so Equitrain can fine-tune LoRA adapters on top of the
    frozen imported MACE-JAX state.
    """

    def __init__(
        self,
        inner_module,
        *,
        rank_fraction: float | int | None = None,
        rank_reduction: float | int | None = None,
        min_rank: int = 1,
        alpha: float | None = None,
    ):
        self._inner = inner_module
        self._graphdef = nnx.graphdef(inner_module)
        param_tree = state_to_pure_dict(nnx.state(inner_module, nnx.Param))
        self.lora_specs = build_lora_spec_map_from_param_tree(
            param_tree,
            rank_fraction=rank_fraction,
            rank_reduction=rank_reduction,
            min_rank=min_rank,
            alpha=alpha,
        )
        self.lora_template = build_lora_template_from_param_tree(
            param_tree,
            self.lora_specs,
        )

    def init(self, *_args, **_kwargs):
        _, state = nnx.split(self._inner)
        base_vars = state_to_pure_dict(state)
        return ensure_lora_params(base_vars, self.lora_template)

    def apply(self, variables, *args, **kwargs):
        base_tree = variables['base_params']
        lora_tree = variables['params']['lora']
        actual_vars = merge_lora_params(base_tree, lora_tree, self.lora_specs)
        outputs, _ = self._graphdef.apply(actual_vars)(*args, **kwargs)
        return outputs


def wrap_with_lora(
    module,
    *,
    rank_fraction: float | int | None = None,
    rank_reduction: float | int | None = None,
    min_rank: int = 1,
    alpha: float | None = None,
):
    return LoRAFineTuneModule(
        module,
        rank_fraction=rank_fraction,
        rank_reduction=rank_reduction,
        min_rank=min_rank,
        alpha=alpha,
    )


__all__ = [
    'LoRASpec',
    'LoRAFineTuneModule',
    'build_lora_spec_map',
    'build_lora_spec_map_from_param_tree',
    'build_lora_template',
    'build_lora_template_from_param_tree',
    'ensure_lora_params',
    'merge_lora_params',
    'wrap_with_lora',
]
