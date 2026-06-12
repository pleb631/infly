from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from infly.core.contracts import InferenceRequest
from infly.core.errors import ErrorCode, PlatformError
from infly.core.models import ModelDefinition
from infly.runtime.model_loader import load_model
from infly.runtime.registry import ModelRegistry
from infly.runtime.service import InferenceService


@pytest.fixture
def log() -> Mock:
    return Mock()


def test_loader_builds_model_with_module_dict_and_kwargs(log: Mock) -> None:
    registry = ModelRegistry()
    definition = ModelDefinition(
        model_name="echo",
        class_path="tests.support.fake_models:EchoModel",
        module_dict={"gpu": 0},
        kwargs={"backend": "onnx"},
    )
    registry.add(definition)

    loaded = load_model(registry.get("echo"), log)

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
        ("not_a_module_and_class", ErrorCode.INVALID_INPUT, "class_path"),
        (":ClassName", ErrorCode.INVALID_INPUT, "class_path"),
        ("module:", ErrorCode.INVALID_INPUT, "class_path"),
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
    log: Mock,
) -> None:
    definition = ModelDefinition(
        model_name="bad",
        class_path=class_path,
    )
    with pytest.raises(PlatformError) as exc:
        load_model(definition, log)
    assert exc.value.code == expected_code
    assert expected_fragment in str(exc.value)


def test_load_model_reports_missing_internal_dependency_as_internal_error(
    log: Mock,
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
        load_model(definition, log)

    assert exc.value.code == ErrorCode.INTERNAL_ERROR
    assert "missing_dependency" in str(exc.value)


def test_model_definition_rejects_reserved_worker_context() -> None:
    with pytest.raises(ValidationError, match="worker_context"):
        ModelDefinition(
            model_name="echo",
            class_path="tests.support.fake_models:EchoModel",
            module_dict={"worker_context": {}},
        )


def test_inference_service_reloads_replaced_model_definition(log: Mock) -> None:
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
    service = InferenceService(registry, log)

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
