from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_RANGE_RE = re.compile(r'^(\d*)\s*[-:]\s*(\d*)$')
_COLLAPSED_MODULE_LISTS = {'readouts'}
_MACE_BLOCK_ANCHORS = ('interactions', 'products')


def semantic_layer_name(parameter_name: str, model: Any | None = None) -> str:
    parts = parameter_name.split('.')
    if parts and parts[0] == 'model':
        parts = parts[1:]
    if not parts:
        return parameter_name

    top_level = parts[0]
    if (
        model is not None
        and top_level not in _COLLAPSED_MODULE_LISTS
        and len(parts) > 1
        and parts[1].isdigit()
        and _is_module_list(_top_level_module(model, top_level))
    ):
        return f'{top_level}.{parts[1]}'
    if top_level in {'interactions', 'products'} and len(parts) > 1:
        return f'{top_level}.{parts[1]}'
    if top_level == 'readouts':
        return top_level
    return top_level


def infer_semantic_layer_names(
    parameter_names: Iterable[str],
    model: Any | None = None,
) -> tuple[str, ...]:
    layer_names = []
    for name in parameter_names:
        layer_name = semantic_layer_name(name, model=model)
        if layer_name not in layer_names:
            layer_names.append(layer_name)
    if model is not None:
        return _model_forward_layer_order(layer_names, model)
    return _mace_forward_layer_order(layer_names)


def _model_forward_layer_order(layer_names: list[str], model: Any) -> tuple[str, ...]:
    block_group_names = _mace_block_group_names(layer_names, model)
    if not block_group_names:
        return _mace_forward_layer_order(layer_names)

    block_group_set = set(block_group_names)
    layer_set = set(layer_names)
    ordered = []
    added = set()

    for layer_name in layer_names:
        if _indexed_layer_group(layer_name) in block_group_set:
            break
        ordered.append(layer_name)
        added.add(layer_name)

    for index in _indexed_layer_indices(layer_names, block_group_names):
        for group_name in block_group_names:
            layer_name = f'{group_name}.{index}'
            if layer_name in layer_set and layer_name not in added:
                ordered.append(layer_name)
                added.add(layer_name)

    for layer_name in layer_names:
        if layer_name not in added:
            ordered.append(layer_name)
            added.add(layer_name)

    return tuple(ordered)


def _mace_block_group_names(layer_names: list[str], model: Any) -> tuple[str, ...]:
    available_groups = {
        group_name
        for layer_name in layer_names
        if (group_name := _indexed_layer_group(layer_name)) is not None
    }
    if not set(_MACE_BLOCK_ANCHORS).issubset(available_groups):
        return ()

    anchor_modules = [_top_level_module(model, name) for name in _MACE_BLOCK_ANCHORS]
    if not all(_is_module_list(module) for module in anchor_modules):
        return ()
    anchor_length = _module_length(anchor_modules[0])

    ordered_groups = list(_MACE_BLOCK_ANCHORS)
    for child_name, child in _named_children(model):
        if (
            child_name in ordered_groups
            or child_name in _COLLAPSED_MODULE_LISTS
            or child_name not in available_groups
            or not _is_module_list(child)
            or _module_length(child) != anchor_length
        ):
            continue
        ordered_groups.append(child_name)
    return tuple(ordered_groups)


def _indexed_layer_indices(
    layer_names: Iterable[str],
    group_names: tuple[str, ...],
) -> tuple[str, ...]:
    indices: set[str] = set()
    for layer_name in layer_names:
        for group_name in group_names:
            index = _indexed_layer_suffix(layer_name, group_name)
            if index is not None:
                indices.add(index)
    return tuple(sorted(indices, key=int))


def _indexed_layer_group(layer_name: str) -> str | None:
    group_name, sep, suffix = layer_name.rpartition('.')
    if sep and suffix.isdigit():
        return group_name
    return None


def _top_level_module(model: Any, name: str) -> Any | None:
    modules = getattr(model, '_modules', {})
    return modules.get(name)


def _named_children(model: Any):
    named_children = getattr(model, 'named_children', None)
    if named_children is None:
        return ()
    return tuple(named_children())


def _is_module_list(module: Any) -> bool:
    if module is None:
        return False
    try:
        import torch
    except ModuleNotFoundError:
        return module.__class__.__name__ == 'ModuleList'
    return isinstance(module, torch.nn.ModuleList)


def _module_length(module: Any) -> int | None:
    try:
        return len(module)
    except TypeError:
        return None


def _mace_forward_layer_order(layer_names: list[str]) -> tuple[str, ...]:
    """Return semantic layers in MACE forward order when possible.

    ``named_parameters()`` visits all ``interactions`` before all ``products``
    because those are separate ``ModuleList`` attributes. MACE executes them as
    ``interaction[i]`` followed by ``product[i]``. Keep unrelated layers in their
    original traversal order, but interleave matching interaction/product blocks
    so numeric freeze ranges follow the model execution order.
    """
    layer_set = set(layer_names)
    ordered = []
    added = set()
    for layer_name in layer_names:
        if layer_name in added:
            continue

        interaction_index = _indexed_layer_suffix(layer_name, 'interactions')
        if interaction_index is not None:
            ordered.append(layer_name)
            added.add(layer_name)
            product_name = f'products.{interaction_index}'
            if product_name in layer_set and product_name not in added:
                ordered.append(product_name)
                added.add(product_name)
            continue

        product_index = _indexed_layer_suffix(layer_name, 'products')
        interaction_name = (
            f'interactions.{product_index}' if product_index is not None else None
        )
        if interaction_name is not None and interaction_name in layer_set:
            continue

        ordered.append(layer_name)
        added.add(layer_name)

    return tuple(ordered)


def _indexed_layer_suffix(layer_name: str, prefix: str) -> str | None:
    marker = f'{prefix}.'
    if not layer_name.startswith(marker):
        return None
    suffix = layer_name[len(marker) :]
    return suffix if suffix.isdigit() else None


def semantic_layer_indices(
    parameter_names: Iterable[str],
    layer_names: tuple[str, ...],
    model: Any | None = None,
) -> dict[str, int]:
    layer_index = {name: index for index, name in enumerate(layer_names)}
    return {
        name: layer_index[semantic_layer_name(name, model=model)]
        for name in parameter_names
    }


def parse_layer_selection(selection, layer_names: tuple[str, ...]) -> set[int]:
    if selection is None or selection == '':
        return set()

    if isinstance(selection, int):
        tokens = [str(selection)]
    elif isinstance(selection, str):
        tokens = [token.strip() for token in selection.split(',') if token.strip()]
    else:
        tokens = []
        for item in selection:
            if isinstance(item, str):
                tokens.extend(
                    token.strip() for token in item.split(',') if token.strip()
                )
            else:
                tokens.append(str(item))

    selected: set[int] = set()
    max_index = len(layer_names) - 1
    for token in tokens:
        match = _RANGE_RE.fullmatch(token)
        if match is not None:
            start_s, end_s = match.groups()
            if start_s == '' and end_s == '':
                raise ValueError('Empty fine-tune layer range is not valid.')
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else max_index
            if start > max_index:
                selected.add(start)
                continue
            if start > end:
                raise ValueError(f'Invalid fine-tune layer range {token!r}.')
            selected.update(range(start, end + 1))
            continue

        try:
            selected.add(int(token))
        except ValueError as exc:
            raise ValueError(
                f'Invalid fine-tune layer selector {token!r}; '
                'use e.g. "2-", "1,3", or "2-4".'
            ) from exc

    invalid = sorted(index for index in selected if index < 0 or index > max_index)
    if invalid:
        layers = ', '.join(f'{index}:{name}' for index, name in enumerate(layer_names))
        raise ValueError(
            f'Fine-tune layer selector out of range: {invalid}. Available layers: {layers}'
        )
    return selected


__all__ = [
    'infer_semantic_layer_names',
    'parse_layer_selection',
    'semantic_layer_indices',
    'semantic_layer_name',
]
