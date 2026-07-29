from __future__ import annotations

import re
from collections.abc import Iterable

_RANGE_RE = re.compile(r'^(\d*)\s*[-:]\s*(\d*)$')


def semantic_layer_name(parameter_name: str) -> str:
    parts = parameter_name.split('.')
    if parts and parts[0] == 'model':
        parts = parts[1:]
    if not parts:
        return parameter_name

    top_level = parts[0]
    if top_level in {'interactions', 'products'} and len(parts) > 1:
        return f'{top_level}.{parts[1]}'
    if top_level == 'readouts':
        return top_level
    return top_level


def infer_semantic_layer_names(parameter_names: Iterable[str]) -> tuple[str, ...]:
    layer_names = []
    for name in parameter_names:
        layer_name = semantic_layer_name(name)
        if layer_name not in layer_names:
            layer_names.append(layer_name)
    return tuple(layer_names)


def semantic_layer_indices(
    parameter_names: Iterable[str],
    layer_names: tuple[str, ...],
) -> dict[str, int]:
    layer_index = {name: index for index, name in enumerate(layer_names)}
    return {name: layer_index[semantic_layer_name(name)] for name in parameter_names}


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
