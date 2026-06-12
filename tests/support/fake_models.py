class EchoModel:
    def __init__(self, module_dict: dict, **kwargs) -> None:
        self.module_dict = module_dict
        self.kwargs = kwargs

    def predict(self, payload: dict) -> dict:
        return {
            "echo": payload["text"],
            "gpu": self.module_dict.get("gpu"),
            "backend": self.kwargs.get("backend"),
            "device": self.kwargs.get("device"),
        }


class FailingModel:
    """Model that raises during construction to simulate startup failure."""

    def __init__(self, module_dict: dict, **kwargs) -> None:
        raise RuntimeError("intentional startup failure")


class SlowModel:
    def __init__(self, module_dict: dict, **kwargs) -> None:
        self.delay_seconds = kwargs.get("delay_seconds", 0.2)

    def predict(self, payload: dict) -> dict:
        import time

        time.sleep(self.delay_seconds)
        return {"echo": payload["text"]}


class CountingModel:
    instances = 0

    def __init__(self, module_dict: dict, **kwargs) -> None:
        type(self).instances += 1
        self.instance_number = type(self).instances

    def predict(self, payload: dict) -> dict:
        return payload


class ContextModel:
    def __init__(self, module_dict: dict, **kwargs) -> None:
        self.module_dict = module_dict

    def predict(self, payload: dict) -> dict:
        import os

        return {
            "payload": payload,
            "worker_context": self.module_dict["worker_context"],
            "environment_device": os.environ.get("INFLY_DEVICE"),
            "custom_environment": os.environ.get("INFLY_TEST_ENV"),
        }


class SlowInitModel:
    def __init__(self, module_dict: dict, **kwargs) -> None:
        import time

        time.sleep(kwargs.get("delay_seconds", 1))

    def predict(self, payload: dict) -> dict:
        return payload


class RaisingPredictModel:
    def __init__(self, module_dict: dict, **kwargs) -> None:
        pass

    def predict(self, payload: dict) -> dict:
        raise RuntimeError("intentional prediction failure")
