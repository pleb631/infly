from infly.core.errors import ErrorCode, PlatformError
from infly.core.handlers import HandlerDefinition
from infly.runtime.log import get_logger

log = get_logger()


class HandlerRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, HandlerDefinition] = {}

    def add(self, definition: HandlerDefinition) -> None:
        replaced = definition.handler_name in self._definitions
        self._definitions[definition.handler_name] = definition
        log.info(
            "handler_registered handler=%s replaced=%s",
            definition.handler_name,
            replaced,
        )

    def get(self, handler_name: str) -> HandlerDefinition:
        if handler_name not in self._definitions:
            log.warning("handler_lookup_failed handler=%s", handler_name)
            raise PlatformError(
                ErrorCode.HANDLER_NOT_FOUND,
                f"Handler '{handler_name}' not found in registry.",
            )
        log.debug("handler_lookup_completed handler=%s", handler_name)
        return self._definitions[handler_name]

    def list(self, handler_name: str | None = None) -> list[HandlerDefinition]:
        definitions = list(self._definitions.values())
        if handler_name is not None:
            definitions = [item for item in definitions if item.handler_name == handler_name]
        definitions = sorted(definitions, key=lambda item: item.handler_name)
        log.debug(
            "handler_list_completed filter=%s count=%s",
            handler_name,
            len(definitions),
        )
        return definitions


__all__ = ["HandlerRegistry"]
