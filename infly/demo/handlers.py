from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from infly.core.errors import ErrorCode, PlatformError


class DemoEchoHandler:
    def __init__(self, init_context: Mapping[str, Any], **init_kwargs: Any) -> None:
        self._init_context = dict(init_context)
        self._prefix = str(init_kwargs.get("prefix", init_context.get("prefix", "")))

    def handle(self, input: Mapping[str, Any]) -> dict[str, Any]:
        runtime_context = self._init_context.get("runtime_context", {})
        text = str(input.get("text", ""))
        return {
            "echo": f"{self._prefix}{text}",
            "runtime_context": runtime_context,
        }


def build_demo_echo_handler(
    init_context: Mapping[str, Any],
    **init_kwargs: Any,
) -> DemoEchoHandler:
    return DemoEchoHandler(init_context, **init_kwargs)


class DemoUnavailableHandler:
    def __init__(self, init_context: Mapping[str, Any], **init_kwargs: Any) -> None:
        self._message = str(init_kwargs.get("message", "demo worker unavailable"))

    def handle(self, input: Mapping[str, Any]) -> Mapping[str, Any]:
        raise PlatformError(ErrorCode.WORKER_UNAVAILABLE, self._message)


def build_demo_unavailable_handler(
    init_context: Mapping[str, Any],
    **init_kwargs: Any,
) -> DemoUnavailableHandler:
    return DemoUnavailableHandler(init_context, **init_kwargs)


__all__ = [
    "DemoEchoHandler",
    "DemoUnavailableHandler",
    "build_demo_echo_handler",
    "build_demo_unavailable_handler",
]
