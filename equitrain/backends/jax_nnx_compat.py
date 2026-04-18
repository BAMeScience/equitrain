from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mace_jax.nnx_config import ConfigDict
from mace_jax.nnx_utils import align_layout_config


def pure_to_serializable_dict(values: Any) -> Any:
    """Convert a pure pytree into a msgpack-safe structure.

    ``mace-jax`` dropped the older ``pure_to_serializable_dict`` helper in favor of
    state-based utilities. ``equitrain`` still serializes already-pure parameter and
    optimizer pytrees, so keep that behavior locally and only normalize
    ``ConfigDict``/mapping containers into plain Python containers.
    """

    def _convert(obj: Any) -> Any:
        if isinstance(obj, ConfigDict):
            return {key: _convert(val) for key, val in obj.items()}
        if isinstance(obj, Mapping):
            return {key: _convert(val) for key, val in obj.items()}
        if isinstance(obj, tuple) and hasattr(obj, '_fields'):
            return type(obj)(*(_convert(val) for val in obj))
        if isinstance(obj, tuple):
            return tuple(_convert(val) for val in obj)
        if isinstance(obj, list):
            return [_convert(val) for val in obj]
        return obj

    return _convert(values)


def normalize_pure_dict(template: Any, values: Any) -> Any:
    """Restore a serialized pure pytree against a template.

    The current ``mace-jax`` loading path aligns layout config leaves against a
    serializable template before restoring state. Mirror that behavior here for
    ``equitrain``'s pure-parameter checkpoints.
    """

    serializable_template = pure_to_serializable_dict(template)
    aligned = align_layout_config(values, serializable_template)
    return _restore_template_types(template, aligned)


def _restore_template_types(template: Any, values: Any) -> Any:
    if isinstance(template, ConfigDict):
        if isinstance(values, ConfigDict):
            return values
        if isinstance(values, Mapping):
            return ConfigDict(
                {
                    key: _restore_template_types(template.get(key), val)
                    for key, val in values.items()
                }
            )
        return values

    if isinstance(template, Mapping) and isinstance(values, Mapping):
        return {
            resolved_key: _restore_template_types(template.get(resolved_key), val)
            for key, val in values.items()
            for resolved_key in (_resolve_mapping_key(template, key),)
        }

    if isinstance(template, tuple) and hasattr(template, '_fields'):
        if isinstance(values, (tuple, list)):
            restored = [
                _restore_template_types(
                    template[idx] if idx < len(template) else None,
                    val,
                )
                for idx, val in enumerate(values)
            ]
            return type(template)(*restored)
        return values

    if isinstance(template, tuple) and isinstance(values, (tuple, list)):
        return tuple(
            _restore_template_types(
                template[idx] if idx < len(template) else None,
                val,
            )
            for idx, val in enumerate(values)
        )

    if isinstance(template, list) and isinstance(values, (tuple, list)):
        return [
            _restore_template_types(
                template[idx] if idx < len(template) else None,
                val,
            )
            for idx, val in enumerate(values)
        ]

    return values


def _resolve_mapping_key(template: Mapping, key: Any) -> Any:
    if key in template:
        return key

    if isinstance(key, str):
        try:
            int_key = int(key)
        except Exception:
            int_key = None
        if int_key is not None and int_key in template:
            return int_key

    key_text = str(key)
    for template_key in template.keys():
        if str(template_key) == key_text:
            return template_key
    return key


__all__ = ['normalize_pure_dict', 'pure_to_serializable_dict']
