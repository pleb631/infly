from collections.abc import Mapping
from typing import Any


class EchoHandler:
    def __init__(self, init_context: Mapping[str, Any], **init_kwargs: Any) -> None:
        self.init_context = init_context
        self.init_kwargs = init_kwargs

    def handle(self, input: Mapping[str, Any]) -> dict[str, Any]:
        prefix = str(self.init_context.get("prefix", ""))
        return {"echo": f"{prefix}{input['text']}"}


def build_echo_handler(
    init_context: Mapping[str, Any],
    **init_kwargs: Any,
) -> EchoHandler:
    return EchoHandler(init_context, **init_kwargs)


NOT_CALLABLE_SYMBOL = {"kind": "not-callable"}


def build_invalid_handler(
    init_context: Mapping[str, Any],
    **init_kwargs: Any,
) -> object:
    return {"init_context": dict(init_context), "init_kwargs": dict(init_kwargs)}


class FailingHandler:
    """Handler that raises during construction to simulate startup failure."""

    def __init__(self, init_context: Mapping[str, Any], **init_kwargs: Any) -> None:
        raise RuntimeError("intentional startup failure")


class SlowHandler:
    def __init__(self, init_context: Mapping[str, Any], **init_kwargs: Any) -> None:
        self.delay_seconds = init_kwargs.get("delay_seconds", 0.2)

    def handle(self, input: Mapping[str, Any]) -> dict[str, Any]:
        import time

        time.sleep(self.delay_seconds)
        return {"echo": input["text"]}


class CountingHandler:
    instances = 0

    def __init__(self, init_context: Mapping[str, Any], **init_kwargs: Any) -> None:
        type(self).instances += 1
        self.instance_number = type(self).instances

    def handle(self, input: Mapping[str, Any]) -> Mapping[str, Any]:
        return input


class ContextHandler:
    def __init__(self, init_context: Mapping[str, Any], **init_kwargs: Any) -> None:
        self.init_context = init_context

    def handle(self, input: Mapping[str, Any]) -> dict[str, Any]:
        import os

        runtime_context = self.init_context.get(
            "runtime_context",
            self.init_context.get("worker_context"),
        )

        return {
            "input": input,
            "runtime_context": runtime_context,
            "worker_context": runtime_context,
            "environment_device": os.environ.get("INFLY_DEVICE"),
            "custom_environment": os.environ.get("INFLY_TEST_ENV"),
        }


class SlowInitHandler:
    def __init__(self, init_context: Mapping[str, Any], **init_kwargs: Any) -> None:
        import time

        time.sleep(init_kwargs.get("delay_seconds", 1))

    def handle(self, input: Mapping[str, Any]) -> Mapping[str, Any]:
        return input


class RaisingHandler:
    def __init__(self, init_context: Mapping[str, Any], **init_kwargs: Any) -> None:
        pass

    def handle(self, input: Mapping[str, Any]) -> dict[str, Any]:
        raise RuntimeError("intentional prediction failure")


