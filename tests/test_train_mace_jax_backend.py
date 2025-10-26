from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
from flax import serialization

# Ensure the sibling mace-jax repository is importable when tests run in-tree
_MACE_JAX_REPO = Path(__file__).resolve().parents[2] / 'mace-jax'
if _MACE_JAX_REPO.exists():
    sys.path.insert(0, str(_MACE_JAX_REPO))


def test_train_mace_jax_backend(tmp_path):
    """
    End-to-end smoke test for the JAX backend using the mace-jax conversion
    helpers. The test downloads a Torch foundation model, converts it to a
    JAX parameter bundle, and runs a short training loop through Equitrain's
    JAX backend.
    """

    pytest.importorskip('jax')
    pytest.importorskip('mace_jax')

    from mace.tools.scripts_utils import extract_config_mace_model  # noqa: PLC0415
    from mace_jax.cli import mace_torch2jax  # noqa: PLC0415

    import equitrain  # noqa: PLC0415
    from equitrain.train import train as run_train  # noqa: PLC0415

    # Download a reference Torch foundation model and convert it to JAX
    try:
        torch_model = mace_torch2jax._load_torch_model_from_foundations('mp', 'small')
    except Exception as exc:  # pragma: no cover - network failures should skip
        pytest.skip(f'Unable to download foundation model: {exc}')

    config = extract_config_mace_model(torch_model)
    if 'error' in config:  # pragma: no cover - unexpected metadata issues
        pytest.skip(
            f'Failed to extract config from foundation model: {config["error"]}'
        )
    config['torch_model_class'] = torch_model.__class__.__name__

    _, variables, _ = mace_torch2jax.convert_model(torch_model, config)

    bundle_dir = tmp_path / 'foundation_small_jax'
    bundle_dir.mkdir()

    (bundle_dir / 'params.msgpack').write_bytes(serialization.to_bytes(variables))
    sanitized_config = _sanitize_config(config)
    (bundle_dir / 'config.json').write_text(json.dumps(sanitized_config))

    args = equitrain.get_args_parser_train().parse_args([])
    args.backend = 'jax'
    args.model = str(bundle_dir)
    data_dir = Path(__file__).with_name('data')
    args.train_file = str(data_dir / 'train.h5')
    args.valid_file = str(data_dir / 'valid.h5')
    args.test_file = str(data_dir / 'train.h5')
    args.output_dir = str(tmp_path / 'jax_backend_output')
    args.epochs = 1
    args.batch_size = 1
    args.lr = 1e-3
    args.verbose = 0
    args.tqdm = False
    args.energy_weight = 1.0
    args.forces_weight = 0.0
    args.stress_weight = 0.0

    run_train(args)

    assert (bundle_dir / 'params.msgpack').exists()
    assert (Path(args.output_dir) / 'jax_params.msgpack').exists()


def _sanitize_config(obj, _seen: set[int] | None = None):
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return '__circular__'
    _seen.add(obj_id)
    try:
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, type):
            return obj.__name__
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, 'detach') and hasattr(obj, 'cpu') and hasattr(obj, 'tolist'):
            try:
                return obj.detach().cpu().tolist()
            except Exception:
                pass
        module_name = getattr(obj.__class__, '__module__', '')
        if module_name.startswith(('e3nn', 'mace')):
            try:
                return str(obj)
            except Exception:
                pass
        if isinstance(obj, Mapping):
            return {
                str(key): _sanitize_config(value, _seen) for key, value in obj.items()
            }
        if isinstance(obj, Sequence) and not isinstance(obj, (bytes, bytearray)):
            return [_sanitize_config(item, _seen) for item in obj]
        return str(obj)
    finally:
        _seen.discard(obj_id)
