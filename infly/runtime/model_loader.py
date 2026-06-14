from importlib import import_module
from typing import cast

from infly.core.errors import ErrorCode, PlatformError
from infly.core.models import ModelDefinition
from infly.core.ports import ModelProtocol

from infly.runtime.log import get_logger
log = get_logger()


def load_model(definition: ModelDefinition) -> ModelProtocol:
    log.info(
        "model_load_started model=%s class_path=%s",
        definition.model_name,
        definition.class_path,
    )
    parts = definition.class_path.split(":")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        log.error(
            "model_load_rejected model=%s reason=malformed_class_path",
            definition.model_name,
        )
        raise PlatformError(
            ErrorCode.INVALID_INPUT,
            f"Malformed class_path: '{definition.class_path}'. Expected format 'module:ClassName'.",
        )
    module_name, class_name = parts
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        log.error(
            "model_load_failed model=%s module=%s error=%s",
            definition.model_name,
            module_name,
            exc,
            exc_info=True,
        )
        missing_name = exc.name or ""
        if missing_name != module_name and not module_name.startswith(
            f"{missing_name}."
        ):
            raise PlatformError(
                ErrorCode.INTERNAL_ERROR,
                f"Module '{module_name}' could not be imported because dependency "
                f"'{missing_name or 'unknown'}' was not found.",
            ) from exc
        raise PlatformError(
            ErrorCode.NOT_FOUND,
            f"Module '{module_name}' not found for class_path '{definition.class_path}'.",
        ) from exc
    try:
        model_class = cast(type[ModelProtocol], getattr(module, class_name))
    except AttributeError as exc:
        log.error(
            "model_load_failed model=%s class=%s error=%s",
            definition.model_name,
            class_name,
            exc,
            exc_info=True,
        )
        raise PlatformError(
            ErrorCode.NOT_FOUND,
            f"Class '{class_name}' not found in module '{module_name}' for class_path '{definition.class_path}'.",
        ) from exc
    try:
        model = model_class(definition.module_dict, **definition.kwargs)
    except Exception as exc:
        log.error(
            "model_load_failed model=%s class_path=%s error=%s",
            definition.model_name,
            definition.class_path,
            exc,
            exc_info=True,
        )
        raise PlatformError(
            ErrorCode.INTERNAL_ERROR,
            f"Model '{definition.model_name}' failed to initialize: {exc}",
        ) from exc
    log.info("model_load_completed model=%s", definition.model_name)
    return cast(ModelProtocol, model)


__all__ = ["load_model"]
