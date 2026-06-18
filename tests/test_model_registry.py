from unittest.mock import Mock

import pytest

from infly.core.contracts import InferenceRequest
from infly.core.errors import ErrorCode, PlatformError
from infly.core.models import ModelDefinition
from infly.runtime.model_loader import load_model
from infly.runtime.registry import ModelRegistry
from infly.runtime.service import InferenceService


def test_loader_builds_model_with_module_dict_and_kwargs() -> None:
    registry = ModelRegistry()
    definition = ModelDefinition(
        model_name="echo",
        class_path="tests.support.fake_models:EchoModel",
        module_dict={"gpu": 0},
        kwargs={"backend": "onnx"},
    )
    registry.add(definition)

    loaded = load_model(registry.get("echo"))

    assert loaded.module_dict == {"gpu": 0}
    assert loaded.kwargs == {"backend": "onnx"}


def test_model_registry_get_missing_raises_platform_error() -> None:
    registry = ModelRegistry()
    with pytest.raises(PlatformError) as exc:
        registry.get("missing")
    assert exc.value.code == ErrorCode.MODEL_NOT_FOUND
    assert "missing" in str(exc.value)


def test_model_registry_lists_definitions_sorted_and_replaces_duplicates() -> None:
    registry = ModelRegistry()
    registry.add(
        ModelDefinition(
            model_name="second",
            class_path="tests.support.fake_models:EchoModel",
        )
    )
    registry.add(
        ModelDefinition(
            model_name="first",
            class_path="tests.support.fake_models:EchoModel",
        ),
    )
    replacement = ModelDefinition(
        model_name="second",
        class_path="tests.support.fake_models:CountingModel",
    )
    registry.add(replacement)

    assert [definition.model_name for definition in registry.list()] == [
        "first",
        "second",
    ]
    assert registry.get("second") is replacement


def test_model_registry_list_can_filter_by_name() -> None:
    registry = ModelRegistry()
    registry.add(
        ModelDefinition(
            model_name="first",
            class_path="tests.support.fake_models:EchoModel",
        )
    )
    registry.add(
        ModelDefinition(
            model_name="second",
            class_path="tests.support.fake_models:EchoModel",
        )
    )

    filtered = registry.list(model_name="second")

    assert [definition.model_name for definition in filtered] == ["second"]


def test_model_definition_cache_key_is_stable_for_equivalent_objects() -> None:
    class ConfigObject:
        def __init__(self, value: str) -> None:
            self.value = value

    first = ModelDefinition(
        model_name="echo",
        class_path="tests.support.fake_models:EchoModel",
        metadata={"config": ConfigObject("same")},
    )
    second = ModelDefinition(
        model_name="echo",
        class_path="tests.support.fake_models:EchoModel",
        metadata={"config": ConfigObject("same")},
    )

    assert first.cache_key == second.cache_key


@pytest.mark.parametrize(
    ("class_path", "expected_code", "expected_fragment"),
    [
        ("not_a_module_and_class", ErrorCode.INVALID_CONFIGURATION, "class_path"),
        (":ClassName", ErrorCode.INVALID_CONFIGURATION, "class_path"),
        ("module:", ErrorCode.INVALID_CONFIGURATION, "class_path"),
        ("notarealmodule:SomeClass", ErrorCode.NOT_FOUND, "notarealmodule"),
        ("tests.support.fake_models:NotAClass", ErrorCode.NOT_FOUND, "NotAClass"),
        (
            "tests.support.fake_models:FailingModel",
            ErrorCode.INTERNAL_ERROR,
            "failed to initialize",
        ),
    ],
)
def test_load_model_invalid_reference_raises_platform_error(
    class_path: str,
    expected_code: ErrorCode,
    expected_fragment: str,
) -> None:
    definition = ModelDefinition(
        model_name="bad",
        class_path=class_path,
    )
    with pytest.raises(PlatformError) as exc:
        load_model(definition)
    assert exc.value.code == expected_code
    assert expected_fragment in str(exc.value)


def test_load_model_reports_missing_internal_dependency_as_internal_error(
    monkeypatch,
) -> None:
    missing_dependency = ModuleNotFoundError(
        "No module named 'missing_dependency'",
        name="missing_dependency",
    )
    monkeypatch.setattr(
        "infly.runtime.model_loader.import_module",
        Mock(side_effect=missing_dependency),
    )
    definition = ModelDefinition(
        model_name="broken",
        class_path="tests.support.fake_models:EchoModel",
    )

    with pytest.raises(PlatformError) as exc:
        load_model(definition)

    assert exc.value.code == ErrorCode.INTERNAL_ERROR
    assert "missing_dependency" in str(exc.value)


def test_model_definition_keeps_reserved_worker_context_value() -> None:
    definition = ModelDefinition(
        model_name="echo",
        class_path="tests.support.fake_models:EchoModel",
        module_dict={"worker_context": {}},
    )
    assert definition.module_dict == {"worker_context": {}}


def test_inference_service_reloads_replaced_model_definition() -> None:
    from tests.support.fake_models import CountingModel

    CountingModel.instances = 0
    registry = ModelRegistry()
    registry.add(
        ModelDefinition(
            model_name="echo",
            class_path="tests.support.fake_models:CountingModel",
            kwargs={"version": 1},
        )
    )
    service = InferenceService(registry)

    first = service.predict(
        InferenceRequest(
            request_id="req-1",
            model_name="echo",
            payload={"text": "one"},
            caller="test",
        )
    )
    assert first.request_id == "req-1"
    assert CountingModel.instances == 1

    registry.add(
        ModelDefinition(
            model_name="echo",
            class_path="tests.support.fake_models:CountingModel",
            kwargs={"version": 2},
        )
    )
    second = service.predict(
        InferenceRequest(
            request_id="req-2",
            model_name="echo",
            payload={"text": "two"},
            caller="test",
        )
    )

    assert second.request_id == "req-2"
    assert CountingModel.instances == 2


def test_inference_service_uses_default_log_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from infly.runtime.log import ContextFilter, log_context
    import logging

    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    service_logger = logging.getLogger("infly.runtime.service")
    model_loader_logger = logging.getLogger("infly.runtime.model_loader")
    original_service_level = service_logger.level
    original_model_loader_level = model_loader_logger.level
    root.handlers = [caplog.handler]
    root.setLevel(logging.DEBUG)
    service_logger.setLevel(logging.DEBUG)
    model_loader_logger.setLevel(logging.INFO)
    caplog.handler.addFilter(ContextFilter())
    caplog.set_level(logging.DEBUG)
    try:
        registry = ModelRegistry()
        registry.add(
            ModelDefinition(
                model_name="echo",
                class_path="tests.support.fake_models:EchoModel",
            )
        )
        service = InferenceService(registry)

        with log_context():
            preload_result = service.preload()
            result = service.predict(
                InferenceRequest(
                    request_id="req-1",
                    model_name="echo",
                    payload={"text": "hello"},
                    caller="test",
                )
            )

        assert preload_result is None
        assert result.request_id == "req-1"
        assert any(
            record.message.startswith("model_preload_started")
            and record.log_category == ""
            and record.log_name == "infly"
            for record in caplog.records
        )
        assert any(
            record.message.startswith("model_load_started")
            and record.log_category == ""
            and record.log_name == "infly"
            for record in caplog.records
        )
        assert any(
            record.message.startswith("inference_started")
            for record in caplog.records
        )
        assert any(
            record.message.startswith("inference_completed")
            for record in caplog.records
        )
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)
        service_logger.setLevel(original_service_level)
        model_loader_logger.setLevel(original_model_loader_level)
