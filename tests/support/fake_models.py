from collections.abc import Mapping
from typing import Any


class EchoModel:
    def __init__(self, module_dict: Mapping[str, Any], **kwargs: Any) -> None:
        self.module_dict = module_dict
        self.kwargs = kwargs

    def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "echo": payload["text"],
            "gpu": self.module_dict.get("gpu"),
            "backend": self.kwargs.get("backend"),
            "device": self.kwargs.get("device"),
        }


def build_echo_model(module_dict: Mapping[str, Any], **kwargs: Any) -> EchoModel:
    return EchoModel(module_dict, **kwargs)


NOT_CALLABLE_SYMBOL = {"kind": "not-callable"}


def build_invalid_model(
    module_dict: Mapping[str, Any],
    **kwargs: Any,
) -> object:
    return {"module_dict": dict(module_dict), "kwargs": dict(kwargs)}


class FailingModel:
    """Model that raises during construction to simulate startup failure."""

    def __init__(self, module_dict: Mapping[str, Any], **kwargs: Any) -> None:
        raise RuntimeError("intentional startup failure")


class SlowModel:
    def __init__(self, module_dict: Mapping[str, Any], **kwargs: Any) -> None:
        self.delay_seconds = kwargs.get("delay_seconds", 0.2)

    def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        import time

        time.sleep(self.delay_seconds)
        return {"echo": payload["text"]}


class CountingModel:
    instances = 0

    def __init__(self, module_dict: Mapping[str, Any], **kwargs: Any) -> None:
        type(self).instances += 1
        self.instance_number = type(self).instances

    def predict(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return payload


class ContextModel:
    def __init__(self, module_dict: Mapping[str, Any], **kwargs: Any) -> None:
        self.module_dict = module_dict

    def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        import os

        return {
            "payload": payload,
            "worker_context": self.module_dict["worker_context"],
            "environment_device": os.environ.get("INFLY_DEVICE"),
            "custom_environment": os.environ.get("INFLY_TEST_ENV"),
        }


class SlowInitModel:
    def __init__(self, module_dict: Mapping[str, Any], **kwargs: Any) -> None:
        import time

        time.sleep(kwargs.get("delay_seconds", 1))

    def predict(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return payload


class RaisingPredictModel:
    def __init__(self, module_dict: Mapping[str, Any], **kwargs: Any) -> None:
        pass

    def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        raise RuntimeError("intentional prediction failure")
