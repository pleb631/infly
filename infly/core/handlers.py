import base64
import json
from collections import ChainMap
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Self


@dataclass(slots=True)
class HandlerDefinition:
    handler_name: str
    entrypoint: str
    init_context: Mapping[str, Any] = field(default_factory=dict)
    init_kwargs: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    reuse_mode: Literal["worker_cached", "task_transient"] = "worker_cached"
    _allow_reserved_runtime_context: InitVar[bool] = False

    def __post_init__(self, _allow_reserved_runtime_context: bool) -> None:
        if "runtime_context" in self.init_context and not _allow_reserved_runtime_context:
            raise ValueError("init_context key 'runtime_context' is reserved")
        if self.reuse_mode not in {"worker_cached", "task_transient"}:
            raise ValueError("reuse_mode must be one of 'worker_cached' or 'task_transient'")

    @classmethod
    def with_runtime_context(
        cls,
        definition: Self,
        *,
        runtime_context: Mapping[str, Any],
    ) -> Self:
        return cls(
            handler_name=definition.handler_name,
            entrypoint=definition.entrypoint,
            init_context=ChainMap(
                {"runtime_context": dict(runtime_context)},
                dict(definition.init_context),
            ),
            init_kwargs=definition.init_kwargs,
            metadata=definition.metadata,
            reuse_mode=definition.reuse_mode,
            _allow_reserved_runtime_context=True,
        )

    @property
    def cache_key(self) -> str:
        normalized = _normalize_cache_value(
            {
                "handler_name": self.handler_name,
                "entrypoint": self.entrypoint,
                "init_context": self.init_context,
                "init_kwargs": self.init_kwargs,
                "metadata": self.metadata,
                "reuse_mode": self.reuse_mode,
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
        value_cls = type(value)
        return {
            "__enum__": f"{value_cls.__module__}.{value_cls.__qualname__}",
            "value": _normalize_cache_value(value.value, _seen=_seen),
        }

    if isinstance(value, Path):
        return {"__path__": str(value)}

    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}

    if isinstance(value, bytearray):
        return {"__bytes__": base64.b64encode(bytes(value)).decode("ascii")}

    if is_dataclass(value) and not isinstance(value, type):
        value_cls = type(value)
        return {
            "__dataclass__": f"{value_cls.__module__}.{value_cls.__qualname__}",
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
            normalized_items = [_normalize_cache_value(item, _seen=_seen) for item in value]
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
            value_cls = type(value)
            return {
                "__object__": (f"{value_cls.__module__}.{value_cls.__qualname__}"),
                "state": _normalize_cache_value(vars(value), _seen=_seen),
            }
        finally:
            _seen.discard(value_id)

    value_cls = type(value)
    raise TypeError(
        f"Unsupported cache_key value type: {value_cls.__module__}.{value_cls.__qualname__}"
    )


__all__ = ["HandlerDefinition"]
