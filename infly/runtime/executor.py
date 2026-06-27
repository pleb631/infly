import threading

from infly.core.contracts import TaskRequest, TaskResult
from infly.core.handlers import HandlerDefinition
from infly.core.ports import HandlerProtocol
from infly.runtime.handler_loader import load_handler
from infly.runtime.log import get_logger
from infly.runtime.registry import HandlerRegistry

log = get_logger()


class HandlerExecutor:
    def __init__(self, registry: HandlerRegistry) -> None:
        self._registry = registry
        self._instances: dict[str, HandlerProtocol] = {}
        self._active_keys: dict[str, str] = {}
        self._instances_lock = threading.Lock()

    def execute(self, request: TaskRequest) -> TaskResult:
        log.debug(
            "handler_execution_started task_key=%s handler=%s caller=%s",
            request.task_key,
            request.handler_name,
            request.caller,
        )
        definition = self._registry.get(request.handler_name)
        if definition.reuse_mode == "task_transient":
            handler = self._load_transient(definition)
        else:
            handler = self._get_or_load(definition.handler_name)

        output = handler.handle(request.input)

        log.debug(
            "handler_execution_completed task_key=%s handler=%s",
            request.task_key,
            request.handler_name,
        )
        return TaskResult(
            task_key=request.task_key,
            output=output,
            diagnostics={
                "handler_name": definition.handler_name,
                "caller": request.caller,
            },
        )

    def preload(self) -> None:
        definitions = self._registry.list()
        log.info("handler_preload_started count=%s", len(definitions))
        for definition in definitions:
            if definition.reuse_mode == "worker_cached":
                self._get_or_load(definition.handler_name)
        log.info("handler_preload_completed count=%s", len(definitions))

    def _get_or_load(self, handler_name: str) -> HandlerProtocol:
        definition = self._registry.get(handler_name)
        cache_key = definition.cache_key
        cached = self._instances.get(cache_key)
        if cached is not None:
            self._active_keys[handler_name] = cache_key
            return cached

        with self._instances_lock:
            definition = self._registry.get(handler_name)
            cache_key = definition.cache_key
            cached = self._instances.get(cache_key)
            if cached is None:
                handler = load_handler(definition)
                self._instances[cache_key] = handler
                previous_key = self._active_keys.get(handler_name)
                self._active_keys[handler_name] = cache_key
                if previous_key is not None and previous_key != cache_key:
                    self._instances.pop(previous_key, None)
            else:
                handler = cached
                self._active_keys[handler_name] = cache_key
                log.debug("handler_cache_hit handler=%s", handler_name)
        return handler

    def _load_transient(self, definition: HandlerDefinition) -> HandlerProtocol:
        with self._instances_lock:
            previous_key = self._active_keys.pop(definition.handler_name, None)
            if previous_key is not None:
                self._instances.pop(previous_key, None)
        return load_handler(definition)



__all__ = ["HandlerExecutor"]

