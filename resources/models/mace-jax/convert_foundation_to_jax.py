"""
Example script that downloads a Torch MACE foundation model, converts it to a
MACE-JAX module, and saves the resulting parameters to disk. Requires both the
``mace`` and ``mace-jax`` Python packages (and their optional cuequivariance
extensions) to be installed in the active environment.

This mirrors the companion resources/models/mace/mace-initial-model.py script,
but demonstrates the cross-framework conversion workflow. Refer to the
``mace_jax`` repository for additional usage patterns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flax import serialization
from mace.tools.scripts_utils import extract_config_mace_model
from mace_jax.cli import mace_torch2jax

# Torch 2.6 tightened the default pickling policy. The foundation checkpoints
# use a ``slice`` object, so we allowlist it explicitly.
try:  # pragma: no cover - defensive import for older torch versions
    from torch.serialization import add_safe_globals
except ImportError:  # pragma: no cover
    add_safe_globals = None

if callable(add_safe_globals):  # pragma: no cover - guard for torch<2.6
    add_safe_globals([slice])


def _sanitize_config(obj: Any) -> Any:
    """
    Convert nested config structures to JSON-friendly types.
    """

    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.bool_, np.integer)):
        return bool(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [_sanitize_config(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): _sanitize_config(value) for key, value in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'tolist'):
        try:
            return obj.tolist()
        except Exception:  # pragma: no cover - best effort
            pass
    if hasattr(obj, '__name__'):
        return obj.__name__
    return str(obj)


def convert_foundation_model(
    *,
    source: str,
    model: str | None,
    output_dir: Path,
) -> Path:
    """
    Download a Torch foundation model and export its parameters for MACE-JAX.

    Parameters
    ----------
    source:
        Foundation family identifier: ``mp``, ``off``, ``anicc``, ``omol``.
    model:
        Optional model tag (e.g. ``small``, ``large``) passed through to the
        foundation loader.
    output_dir:
        Directory where the converted parameters and sanitized configuration
        will be written. The directory is created if it does not exist.

    Returns
    -------
    Path
        The directory containing the exported files.
    """

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_model = mace_torch2jax._load_torch_model_from_foundations(source, model)
    torch_model = torch_model.float().eval()

    config = extract_config_mace_model(torch_model)
    config['model_wrapper'] = 'mace'
    config['torch_model_class'] = torch_model.__class__.__name__

    _, jax_params, _ = mace_torch2jax.convert_model(torch_model, config)

    params_path = output_dir / 'params.msgpack'
    params_path.write_bytes(serialization.to_bytes(jax_params))

    config_path = output_dir / 'config.json'
    config_path.write_text(json.dumps(_sanitize_config(config), indent=2))

    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Convert a Torch MACE foundation model into MACE-JAX parameters. '
            'The exported files can be passed directly to Equitrain'
            ' (via --model pointing at the output directory) or loaded with '
            'mace_jax.tools.gin_model.'
        )
    )
    parser.add_argument(
        '--source',
        default='mp',
        choices=('mp', 'off', 'anicc', 'omol'),
        help='Foundation family to download (default: mp).',
    )
    parser.add_argument(
        '--model',
        default='small',
        help='Optional model identifier passed to the foundation loader.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('mace_jax_foundation_bundle'),
        help='Directory where the converted parameters will be written.',
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = convert_foundation_model(
        source=args.source,
        model=args.model,
        output_dir=args.output_dir,
    )
    print(f'Exported MACE-JAX bundle to {output_dir}')


if __name__ == '__main__':
    main()
