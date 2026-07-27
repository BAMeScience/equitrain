from __future__ import annotations

import torch


def remap_legacy_model_prefix(state_dict, prefix: str) -> None:
    """Map old fine-tune checkpoint keys from ``model.*`` to ``base_wrapper.model.*``."""
    legacy_prefix = f'{prefix}model.'
    current_prefix = f'{prefix}base_wrapper.model.'

    for key in tuple(state_dict):
        if not str(key).startswith(legacy_prefix):
            continue
        target = current_prefix + str(key)[len(legacy_prefix) :]
        if target not in state_dict:
            state_dict[target] = state_dict[key]
        del state_dict[key]


def uniquify_empty_tensor_storage(state_dict) -> None:
    """Give zero-length tensors distinct storage for Accelerate's safetensors pass."""
    for key, tensor in tuple(state_dict.items()):
        if not isinstance(tensor, torch.Tensor) or tensor.numel() != 0:
            continue
        state_dict[key] = tensor.new_empty((1,))[:0].reshape(tensor.shape)


def _has_complete_storage(tensor: torch.Tensor) -> bool:
    try:
        storage = tensor.untyped_storage()
        storage_ptr = storage.data_ptr()
        storage_size = storage.nbytes()
    except Exception:
        return True
    return (
        tensor.data_ptr() == storage_ptr
        and tensor.numel() * tensor.element_size() == storage_size
    )


def clone_non_complete_tensor_storage(state_dict) -> None:
    """Clone non-empty tensor views so safetensors can inspect state dicts."""
    for key, tensor in tuple(state_dict.items()):
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            continue
        if not _has_complete_storage(tensor):
            state_dict[key] = tensor.detach().clone().contiguous()
