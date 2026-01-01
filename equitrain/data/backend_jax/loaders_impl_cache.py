"""Shared streaming stats cache helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from collections.abc import Sequence
from pathlib import Path

STREAMING_STATS_CACHE_VERSION = 1


def stats_cache_path(dataset_path: Path) -> Path:
    """Return the cache path for streaming stats for a dataset.

    Use a hidden sidecar file so glob patterns like ``dir/*`` (used by Torch MACE)
    do not treat the cache as an HDF5 shard.
    """
    cache_name = f'.{dataset_path.stem}.streamstats.pkl'
    return dataset_path.with_name(cache_name)


def dataset_signature(dataset_path: Path) -> dict[str, float]:
    """Compute a lightweight signature used to invalidate cached stats."""
    stat = dataset_path.stat()
    return {'size': stat.st_size, 'mtime': stat.st_mtime}


def spec_fingerprint(
    spec,
    *,
    r_max: float,
    atomic_numbers: Sequence[int],
    edge_cap: int | None = None,
) -> str:
    """Build a stable fingerprint for dataset spec + packing parameters."""
    payload = {
        'energy_key': spec.energy_key,
        'forces_key': spec.forces_key,
        'stress_key': spec.stress_key,
        'virials_key': spec.virials_key,
        'dipole_key': spec.dipole_key,
        'charges_key': spec.charges_key,
        'path': str(Path(spec.path)),
        'r_max': float(r_max),
        'atomic_numbers': [int(z) for z in atomic_numbers],
        'edge_cap': int(edge_cap) if edge_cap is not None else None,
    }
    encoded = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def stats_payload_from_parts(
    batch_assignments: list[list[int]],
    n_nodes: int,
    n_edges: int,
    n_graphs: int,
) -> dict:
    """Serialize streaming stats to a JSON/pickle-friendly payload."""
    return {
        'batch_assignments': batch_assignments,
        'n_nodes': int(n_nodes),
        'n_edges': int(n_edges),
        'n_graphs': int(n_graphs),
    }


def stats_payload_to_parts(payload: dict) -> tuple[list[list[int]], int, int, int]:
    """Deserialize streaming stats payload back into components."""
    return (
        payload['batch_assignments'],
        int(payload['n_nodes']),
        int(payload['n_edges']),
        int(payload['n_graphs']),
    )


def load_cached_streaming_stats(dataset_path: Path, fingerprint: str) -> dict | None:
    """Load cached streaming stats if they match the current dataset and spec."""
    cache_path = stats_cache_path(dataset_path)
    payload = None
    if not cache_path.exists():
        return None
    try:
        with cache_path.open('rb') as fh:
            payload = pickle.load(fh)
    except Exception as exc:  # pragma: no cover - cache corruption is unexpected
        logging.warning('Failed to load streaming stats cache %s: %s', cache_path, exc)
        return None
    if payload is None:
        return None
    if payload.get('version') != STREAMING_STATS_CACHE_VERSION:
        return None
    dataset_sig = payload.get('dataset_signature')
    if dataset_sig != dataset_signature(dataset_path):
        return None
    if payload.get('spec_fingerprint') != fingerprint:
        return None
    stats_payload = payload.get('stats')
    if not stats_payload:
        return None
    logging.info('Loaded cached streaming stats from %s', cache_path)
    return stats_payload


def store_cached_streaming_stats(
    dataset_path: Path,
    fingerprint: str,
    stats_payload: dict,
) -> None:
    """Persist streaming stats to disk for reuse across runs."""
    cache_path = stats_cache_path(dataset_path)
    payload = {
        'version': STREAMING_STATS_CACHE_VERSION,
        'dataset_signature': dataset_signature(dataset_path),
        'spec_fingerprint': fingerprint,
        'stats': stats_payload,
    }
    try:
        with cache_path.open('wb') as fh:
            pickle.dump(payload, fh)
    except OSError as exc:  # pragma: no cover - filesystem failure
        logging.warning('Failed to write streaming stats cache %s: %s', cache_path, exc)
        return
    logging.debug('Cached streaming stats to %s', cache_path)
