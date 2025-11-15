#!/usr/bin/env python3
"""
Utility script to export a TorchANI model checkpoint that can be used with Equitrain.

Example
-------
    python ani-initial-model.py --variant ANI1x --output ani-initial.model
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import torch


def _resolve_factory() -> dict[str, Callable[[], torch.nn.Module]]:
    try:
        import torchani
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            'TorchANI must be installed to export ANI models. '
            'Install with `pip install torchani`.'
        ) from exc

    available: dict[str, Callable[[], torch.nn.Module]] = {}
    for name in ('ANI1x', 'ANI1ccx', 'ANI2x', 'ANI2ccx'):
        factory = getattr(torchani.models, name, None)
        if callable(factory):
            available[name] = factory

    if not available:
        raise RuntimeError(
            'No TorchANI model factories were found in this installation.'
        )

    return available


def _build_arg_parser(choices: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Export a pre-trained TorchANI model that is compatible with Equitrain.'
    )
    parser.add_argument(
        '--variant',
        default='ANI1x',
        choices=choices,
        help='TorchANI model family to export.',
    )
    parser.add_argument(
        '--output',
        default='ani-initial.model',
        help='Destination filename for the serialized model (default: %(default)s).',
    )
    parser.add_argument(
        '--dtype',
        default='float32',
        choices=('float32', 'float64'),
        help='Floating point precision to store the model weights in.',
    )
    return parser


def _set_default_dtype(dtype: str) -> None:
    if dtype == 'float32':
        torch.set_default_dtype(torch.float32)
    elif dtype == 'float64':
        torch.set_default_dtype(torch.float64)
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f'Unsupported dtype requested: {dtype}')


def main() -> None:
    factories = _resolve_factory()
    parser = _build_arg_parser(sorted(factories))
    args = parser.parse_args()

    _set_default_dtype(args.dtype)

    model_factory = factories[args.variant]
    model = model_factory()
    model = model.to(device=torch.device('cpu'), dtype=torch.get_default_dtype())

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, output_path)

    species = getattr(model, 'species_order', None)
    info_lines = [
        f'Saved TorchANI {args.variant} model to `{output_path}`.',
    ]
    if species is not None:
        info_lines.append(f'Supported species order: {list(species)}')
    print('\n'.join(info_lines))


if __name__ == '__main__':
    main()
