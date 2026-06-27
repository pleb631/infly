from collections.abc import Callable
from importlib import import_module
from typing import Any, cast

from infly.core.errors import ErrorCode, PlatformError
from infly.core.handlers import HandlerDefinition
from infly.core.ports import HandlerFactory, HandlerProtocol
from infly.runtime.log import get_logger

log = get_logger()


def _is_handler_instance(candidate: object) -> bool:
    handle = getattr(candidate, "handle", None)
    return callable(handle)


def load_handler(definition: HandlerDefinition) -> HandlerProtocol:
    log.info(
        "handler_load_started handler=%s entrypoint=%s",
        definition.handler_name,
        definition.entrypoint,
    )
    parts = definition.entrypoint.split(":")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        log.error(
            "handler_load_rejected handler=%s reason=malformed_entrypoint",
            definition.handler_name,
        )
        raise PlatformError(
            ErrorCode.INVALID_CONFIGURATION,
            f"Malformed entrypoint: '{definition.entrypoint}'."
            f" Expected format 'module:SymbolName'.",
        )
    module_name, symbol_name = parts
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        log.error(
            "handler_load_failed handler=%s module=%s error=%s",
            definition.handler_name,
            module_name,
            exc,
            exc_info=True,
        )
        missing_name = exc.name or ""
        if missing_name != module_name and not module_name.startswith(f"{missing_name}."):
            raise PlatformError(
                ErrorCode.INTERNAL_ERROR,
                f"Module '{module_name}' could not be imported because dependency "
                f"'{missing_name or 'unknown'}' was not found.",
            ) from exc
        raise PlatformError(
            ErrorCode.NOT_FOUND,
            f"Module '{module_name}' not found for entrypoint '{definition.entrypoint}'.",
        ) from exc
    try:
        factory = cast(Callable[..., Any], getattr(module, symbol_name))
    except AttributeError as exc:
        log.error(
            "handler_load_failed handler=%s symbol=%s error=%s",
            definition.handler_name,
            symbol_name,
            exc,
            exc_info=True,
        )
        raise PlatformError(
            ErrorCode.NOT_FOUND,
            f"Attribute '{symbol_name}' not found in module '{module_name}' "
            f"for entrypoint '{definition.entrypoint}'.",
        ) from exc
    if not callable(factory):
        log.error(
            "handler_load_rejected handler=%s symbol=%s reason=not_callable",
            definition.handler_name,
            symbol_name,
        )
        raise PlatformError(
            ErrorCode.INVALID_CONFIGURATION,
            f"Attribute '{symbol_name}' in module '{module_name}' for entrypoint "
            f"'{definition.entrypoint}' must be a callable that returns HandlerProtocol.",
        )
    try:
        handler = cast(HandlerFactory, factory)(
            definition.init_context,
            **definition.init_kwargs,
        )
    except Exception as exc:
        log.error(
            "handler_load_failed handler=%s entrypoint=%s error=%s",
            definition.handler_name,
            definition.entrypoint,
            exc,
            exc_info=True,
        )
        raise PlatformError(
            ErrorCode.INTERNAL_ERROR,
            f"Handler '{definition.handler_name}' failed to initialize: {exc}",
        ) from exc
    if not _is_handler_instance(handler):
        log.error(
            "handler_load_rejected handler=%s symbol=%s reason=invalid_handler_instance",
            definition.handler_name,
            symbol_name,
        )
        raise PlatformError(
            ErrorCode.INVALID_CONFIGURATION,
            f"Callable '{symbol_name}' in module '{module_name}' for entrypoint "
            f"'{definition.entrypoint}' must return an object with a callable "
            "handle(input) method.",
        )
    log.info("handler_load_completed handler=%s", definition.handler_name)
    return handler


__all__ = ["load_handler"]
