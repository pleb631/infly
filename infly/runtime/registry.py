from infly.core.errors import ErrorCode, PlatformError
from infly.core.models import ModelDefinition
from infly.runtime.log import get_logger

log = get_logger("registry")

class ModelRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, ModelDefinition] = {}

    def add(self, definition: ModelDefinition) -> None:
        replaced = definition.model_name in self._definitions
        self._definitions[definition.model_name] = definition
        log.info(
            "model_registered model=%s replaced=%s",
            definition.model_name,
            replaced,
        )

    def get(self, model_name: str) -> ModelDefinition:
        if model_name not in self._definitions:
            log.warning("model_lookup_failed model=%s", model_name)
            raise PlatformError(
                ErrorCode.MODEL_NOT_FOUND,
                f"Model '{model_name}' not found in registry.",
            )
        log.debug("model_lookup_completed model=%s", model_name)
        return self._definitions[model_name]

    def list(self, model_name: str | None = None) -> list[ModelDefinition]:
        definitions = list(self._definitions.values())
        if model_name is not None:
            definitions = [item for item in definitions if item.model_name == model_name]
        definitions = sorted(definitions, key=lambda item: item.model_name)
        log.debug(
            "model_list_completed filter=%s count=%s",
            model_name,
            len(definitions),
        )
        return definitions


__all__ = ["ModelRegistry"]
