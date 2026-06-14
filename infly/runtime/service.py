import threading

from infly.core.contracts import InferenceRequest, InferenceResult
from infly.core.errors import ErrorCode, PlatformError
from infly.core.ports import ModelProtocol
from infly.runtime.model_loader import load_model
from infly.runtime.registry import ModelRegistry

from infly.runtime.log import get_logger
log = get_logger()


class InferenceService:
    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._instances: dict[str, ModelProtocol] = {}
        self._active_keys: dict[str, str] = {}
        self._instances_lock = threading.Lock()

    def predict(self, request: InferenceRequest) -> InferenceResult:
        log.debug(
            "inference_started request_id=%s model=%s caller=%s",
            request.request_id,
            request.model_name,
            request.caller,
        )
        definition = self._registry.get(request.model_name)
        model = self._get_or_load(definition.model_key)

        data = model.predict(request.payload)

        log.debug(
            "inference_completed request_id=%s model=%s",
            request.request_id,
            request.model_name,
        )
        return InferenceResult(
            request_id=request.request_id,
            data=data,
            diagnostics={
                "model_key": definition.model_key,
                "caller": request.caller,
            },
        )

    def preload(self) -> None:
        definitions = self._registry.list()
        log.info("model_preload_started count=%s", len(definitions))
        for definition in definitions:
            self._get_or_load(definition.model_key)
        log.info("model_preload_completed count=%s", len(definitions))

    def _get_or_load(self, model_name: str) -> ModelProtocol:
        definition = self._registry.get(model_name)
        cache_key = definition.cache_key
        cached = self._instances.get(cache_key)
        if cached is not None:
            self._active_keys[model_name] = cache_key
            return cached

        with self._instances_lock:
            definition = self._registry.get(model_name)
            cache_key = definition.cache_key
            cached = self._instances.get(cache_key)
            if cached is None:
                model = load_model(definition)
                self._instances[cache_key] = model
                previous_key = self._active_keys.get(model_name)
                self._active_keys[model_name] = cache_key
                if previous_key is not None and previous_key != cache_key:
                    self._instances.pop(previous_key, None)
            else:
                model = cached
                self._active_keys[model_name] = cache_key
                log.debug("model_cache_hit model=%s", model_name)
        return model


__all__ = ["InferenceService"]
