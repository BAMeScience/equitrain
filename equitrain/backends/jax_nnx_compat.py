from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:  # pragma: no cover - optional dependency
    from mace_jax.nnx_config import ConfigDict as _MaceConfigDict
    from mace_jax.nnx_config import ConfigVar as _MaceConfigVar
except ImportError:  # pragma: no cover - exercised in environments without mace-jax
    _MaceConfigDict = None
    _MaceConfigVar = None


def is_config_dict(value: Any) -> bool:
    return _MaceConfigDict is not None and isinstance(value, _MaceConfigDict)


def _make_config_dict(values: Mapping) -> Any:
    if _MaceConfigDict is None:
        return dict(values)
    return _MaceConfigDict(dict(values))


def _mapping_items(value: Any):
    if isinstance(value, Mapping):
        return value.items()
    if is_config_dict(value):
        return value.items()
    return None


def state_to_pure_dict(state: Any) -> dict[str, Any]:
    """Convert an NNX state into a pure dict without requiring mace-jax."""

    from flax import nnx

    def _extract(value: Any) -> Any:
        if _MaceConfigVar is not None and isinstance(value, _MaceConfigVar):
            config_val = value.get_value()
            if isinstance(config_val, dict) and not is_config_dict(config_val):
                return _make_config_dict(config_val)
            return config_val
        if isinstance(value, nnx.Variable):
            return value.get_value()
        return value

    return nnx.to_pure_dict(state, extract_fn=_extract)


def pure_to_serializable_dict(values: Any) -> Any:
    """Convert a pure pytree into a msgpack-safe structure.

    ``mace-jax`` dropped the older ``pure_to_serializable_dict`` helper in favor of
    state-based utilities. ``equitrain`` still serializes already-pure parameter and
    optimizer pytrees, so keep that behavior locally and only normalize
    ``ConfigDict``/mapping containers into plain Python containers.
    """

    def _convert(obj: Any) -> Any:
        if is_config_dict(obj):
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


def _align_layout_config(values: Any, template: Any) -> Any:
    template_items = _mapping_items(template)
    value_items = _mapping_items(values)
    if template_items is not None and value_items is not None:
        template_dict = dict(template_items)
        return {
            resolved_key: _align_layout_config(
                val,
                template_dict.get(resolved_key),
            )
            for key, val in value_items
            for resolved_key in (_resolve_mapping_key(template_dict, key),)
        }

    if isinstance(template, tuple | list) and isinstance(values, tuple | list):
        aligned = [
            _align_layout_config(
                val,
                template[idx] if idx < len(template) else None,
            )
            for idx, val in enumerate(values)
        ]
        return tuple(aligned) if isinstance(values, tuple) else aligned

    return values


def normalize_pure_dict(template: Any, values: Any) -> Any:
    """Restore a serialized pure pytree against a template.

    The current ``mace-jax`` loading path aligns layout config leaves against a
    serializable template before restoring state. Mirror that behavior here for
    ``equitrain``'s pure-parameter checkpoints.
    """

    serializable_template = pure_to_serializable_dict(template)
    aligned = _align_layout_config(values, serializable_template)
    return _restore_template_types(template, aligned)


def _restore_template_types(template: Any, values: Any) -> Any:
    if is_config_dict(template):
        if is_config_dict(values):
            return values
        value_items = _mapping_items(values)
        if value_items is not None:
            return _make_config_dict(
                {
                    key: _restore_template_types(template.get(key), val)
                    for key, val in value_items
                }
            )
        return values

    value_items = _mapping_items(values)
    if isinstance(template, Mapping) and value_items is not None:
        return {
            resolved_key: _restore_template_types(template.get(resolved_key), val)
            for key, val in value_items
            for resolved_key in (_resolve_mapping_key(template, key),)
        }

    if isinstance(template, tuple) and hasattr(template, '_fields'):
        if isinstance(values, tuple | list):
            restored = [
                _restore_template_types(
                    template[idx] if idx < len(template) else None,
                    val,
                )
                for idx, val in enumerate(values)
            ]
            return type(template)(*restored)
        return values

    if isinstance(template, tuple) and isinstance(values, tuple | list):
        return tuple(
            _restore_template_types(
                template[idx] if idx < len(template) else None,
                val,
            )
            for idx, val in enumerate(values)
        )

    if isinstance(template, list) and isinstance(values, tuple | list):
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


__all__ = [
    'is_config_dict',
    'normalize_pure_dict',
    'pure_to_serializable_dict',
    'state_to_pure_dict',
]
