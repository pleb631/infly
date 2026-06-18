import base64
from dataclasses import dataclass, field, fields, is_dataclass
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

@dataclass(slots=True)
class ModelDefinition:
    model_name: str
    class_path: str
    module_dict: Mapping[str, Any] = field(default_factory=dict)
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def model_key(self) -> str:
        return self.model_name

    @property
    def cache_key(self) -> str:
        normalized = _normalize_cache_value(
            {
                "model_name": self.model_name,
                "class_path": self.class_path,
                "module_dict": self.module_dict,
                "kwargs": self.kwargs,
                "metadata": self.metadata,
            }
        )
        return json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


def _normalize_cache_value(value: Any, *, _seen: set[int] | None = None) -> Any:
    if _seen is None:
        _seen = set()

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Enum):
        return {
            "__enum__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "value": _normalize_cache_value(value.value, _seen=_seen),
        }

    if isinstance(value, Path):
        return {"__path__": str(value)}

    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}

    if isinstance(value, bytearray):
        return {"__bytes__": base64.b64encode(bytes(value)).decode("ascii")}

    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "fields": _normalize_cache_value(
                {field.name: getattr(value, field.name) for field in fields(value)},
                _seen=_seen,
            ),
        }

    value_id = id(value)
    if value_id in _seen:
        raise TypeError("cache_key does not support recursive values")

    if isinstance(value, Mapping):
        _seen.add(value_id)
        try:
            return {
                str(key): _normalize_cache_value(item_value, _seen=_seen)
                for key, item_value in sorted(
                    value.items(),
                    key=lambda item: str(item[0]),
                )
            }
        finally:
            _seen.discard(value_id)

    if isinstance(value, (list, tuple)):
        _seen.add(value_id)
        try:
            return [_normalize_cache_value(item, _seen=_seen) for item in value]
        finally:
            _seen.discard(value_id)

    if isinstance(value, (set, frozenset)):
        _seen.add(value_id)
        try:
            normalized_items = [
                _normalize_cache_value(item, _seen=_seen) for item in value
            ]
            return sorted(
                normalized_items,
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            )
        finally:
            _seen.discard(value_id)

    if hasattr(value, "__dict__"):
        _seen.add(value_id)
        try:
            return {
                "__object__": (
                    f"{value.__class__.__module__}.{value.__class__.__qualname__}"
                ),
                "state": _normalize_cache_value(vars(value), _seen=_seen),
            }
        finally:
            _seen.discard(value_id)

    raise TypeError(
        "Unsupported cache_key value type: "
        f"{value.__class__.__module__}.{value.__class__.__qualname__}"
    )


__all__ = ["ModelDefinition"]
