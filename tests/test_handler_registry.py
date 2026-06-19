from unittest.mock import Mock

import pytest

from infly.core.contracts import TaskRequest
from infly.core.errors import ErrorCode, PlatformError
from infly.core.handlers import HandlerDefinition
from infly.runtime.executor import HandlerExecutor
from infly.runtime.handler_loader import load_handler
from infly.runtime.registry import HandlerRegistry


def test_handler_loader_builds_handler_class_with_init_context_and_kwargs() -> None:
    registry = HandlerRegistry()
    definition = HandlerDefinition(
        handler_name="echo",
        entrypoint="tests.support.fake_handlers:EchoHandler",
        init_context={"prefix": "ok:"},
        init_kwargs={"backend": "onnx"},
    )
    registry.add(definition)

    loaded = load_handler(registry.get("echo"))

    assert loaded.init_context == {"prefix": "ok:"}
    assert loaded.init_kwargs == {"backend": "onnx"}


def test_handler_loader_builds_handler_factory_with_init_context_and_kwargs() -> None:
    definition = HandlerDefinition(
        handler_name="echo",
        entrypoint="tests.support.fake_handlers:build_echo_handler",
        init_context={"prefix": "ok:"},
        init_kwargs={"backend": "torch", "device": "cuda:0"},
    )

    loaded = load_handler(definition)

    assert loaded.init_context == {"prefix": "ok:"}
    assert loaded.init_kwargs == {"backend": "torch", "device": "cuda:0"}


def test_handler_registry_get_missing_raises_platform_error() -> None:
    registry = HandlerRegistry()
    with pytest.raises(PlatformError) as exc:
        registry.get("missing")
    assert exc.value.code == ErrorCode.HANDLER_NOT_FOUND
    assert "missing" in str(exc.value)


def test_handler_registry_lists_definitions_sorted_and_replaces_duplicates() -> None:
    registry = HandlerRegistry()
    registry.add(
        HandlerDefinition(
            handler_name="second",
            entrypoint="tests.support.fake_handlers:EchoHandler",
        )
    )
    registry.add(
        HandlerDefinition(
            handler_name="first",
            entrypoint="tests.support.fake_handlers:EchoHandler",
        ),
    )
    replacement = HandlerDefinition(
        handler_name="second",
        entrypoint="tests.support.fake_handlers:CountingHandler",
    )
    registry.add(replacement)

    assert [definition.handler_name for definition in registry.list()] == [
        "first",
        "second",
    ]
    assert registry.get("second") is replacement


def test_handler_registry_list_can_filter_by_name() -> None:
    registry = HandlerRegistry()
    registry.add(
        HandlerDefinition(
            handler_name="first",
            entrypoint="tests.support.fake_handlers:EchoHandler",
        )
    )
    registry.add(
        HandlerDefinition(
            handler_name="second",
            entrypoint="tests.support.fake_handlers:EchoHandler",
        )
    )

    filtered = registry.list(handler_name="second")

    assert [definition.handler_name for definition in filtered] == ["second"]


def test_handler_definition_cache_key_is_stable_for_equivalent_objects() -> None:
    class ConfigObject:
        def __init__(self, value: str) -> None:
            self.value = value

    first = HandlerDefinition(
        handler_name="echo",
        entrypoint="tests.support.fake_handlers:EchoHandler",
        metadata={"config": ConfigObject("same")},
    )
    second = HandlerDefinition(
        handler_name="echo",
        entrypoint="tests.support.fake_handlers:EchoHandler",
        metadata={"config": ConfigObject("same")},
    )

    assert first.cache_key == second.cache_key


@pytest.mark.parametrize(
    ("entrypoint", "expected_code", "expected_fragment"),
    [
        ("not_a_module_and_class", ErrorCode.INVALID_CONFIGURATION, "entrypoint"),
        (":SymbolName", ErrorCode.INVALID_CONFIGURATION, "entrypoint"),
        ("module:", ErrorCode.INVALID_CONFIGURATION, "entrypoint"),
        ("notarealmodule:SomeClass", ErrorCode.NOT_FOUND, "notarealmodule"),
        (
            "tests.support.fake_handlers:NotAClass",
            ErrorCode.NOT_FOUND,
            "Attribute 'NotAClass'",
        ),
        (
            "tests.support.fake_handlers:FailingHandler",
            ErrorCode.INTERNAL_ERROR,
            "failed to initialize",
        ),
    ],
)
def test_load_handler_invalid_reference_raises_platform_error(
    entrypoint: str,
    expected_code: ErrorCode,
    expected_fragment: str,
) -> None:
    definition = HandlerDefinition(
        handler_name="bad",
        entrypoint=entrypoint,
    )
    with pytest.raises(PlatformError) as exc:
        load_handler(definition)
    assert exc.value.code == expected_code
    assert expected_fragment in str(exc.value)


def test_load_handler_rejects_non_callable_symbol() -> None:
    definition = HandlerDefinition(
        handler_name="bad",
        entrypoint="tests.support.fake_handlers:NOT_CALLABLE_SYMBOL",
    )

    with pytest.raises(PlatformError) as exc:
        load_handler(definition)

    assert exc.value.code == ErrorCode.INVALID_CONFIGURATION
    assert "must be a callable" in str(exc.value)


def test_load_handler_rejects_factory_return_without_handle() -> None:
    definition = HandlerDefinition(
        handler_name="bad",
        entrypoint="tests.support.fake_handlers:build_invalid_handler",
        init_context={"gpu": 0},
        init_kwargs={"backend": "onnx"},
    )

    with pytest.raises(PlatformError) as exc:
        load_handler(definition)

    assert exc.value.code == ErrorCode.INVALID_CONFIGURATION
    assert "handle(input)" in str(exc.value)


def test_load_handler_reports_missing_internal_dependency_as_internal_error(
    monkeypatch,
) -> None:
    missing_dependency = ModuleNotFoundError(
        "No module named 'missing_dependency'",
        name="missing_dependency",
    )
    monkeypatch.setattr(
        "infly.runtime.handler_loader.import_module",
        Mock(side_effect=missing_dependency),
    )
    definition = HandlerDefinition(
        handler_name="broken",
        entrypoint="tests.support.fake_handlers:EchoHandler",
    )

    with pytest.raises(PlatformError) as exc:
        load_handler(definition)

    assert exc.value.code == ErrorCode.INTERNAL_ERROR
    assert "missing_dependency" in str(exc.value)


def test_handler_definition_rejects_reserved_runtime_context() -> None:
    with pytest.raises(ValueError, match="runtime_context"):
        HandlerDefinition(
            handler_name="echo",
            entrypoint="tests.support.fake_handlers:EchoHandler",
            init_context={"runtime_context": {}},
        )


def test_handler_definition_allows_runtime_context_injection() -> None:
    definition = HandlerDefinition(
        handler_name="echo",
        entrypoint="tests.support.fake_handlers:EchoHandler",
        init_context={"gpu": 0},
    )

    runtime_definition = HandlerDefinition.with_runtime_context(
        definition,
        runtime_context={"worker_id": "cpu_R0"},
    )

    assert runtime_definition.init_context["runtime_context"] == {
        "worker_id": "cpu_R0"
    }
    assert runtime_definition.init_context["gpu"] == 0
    assert definition.init_context == {"gpu": 0}


def test_handler_definition_cache_key_includes_reuse_mode() -> None:
    cached = HandlerDefinition(
        handler_name="echo",
        entrypoint="tests.support.fake_handlers:EchoHandler",
    )
    transient = HandlerDefinition(
        handler_name="echo",
        entrypoint="tests.support.fake_handlers:EchoHandler",
        reuse_mode="task_transient",
    )

    assert cached.cache_key != transient.cache_key


def test_handler_executor_reloads_replaced_definition() -> None:
    from tests.support.fake_handlers import CountingHandler

    CountingHandler.instances = 0
    registry = HandlerRegistry()
    registry.add(
        HandlerDefinition(
            handler_name="echo",
            entrypoint="tests.support.fake_handlers:CountingHandler",
            init_kwargs={"version": 1},
        )
    )
    executor = HandlerExecutor(registry)

    first = executor.execute(
        TaskRequest(
            task_key="req-1",
            handler_name="echo",
            input={"text": "one"},
            caller="test",
        )
    )
    assert first.task_key == "req-1"
    assert CountingHandler.instances == 1

    registry.add(
        HandlerDefinition(
            handler_name="echo",
            entrypoint="tests.support.fake_handlers:CountingHandler",
            init_kwargs={"version": 2},
        )
    )
    second = executor.execute(
        TaskRequest(
            task_key="req-2",
            handler_name="echo",
            input={"text": "two"},
            caller="test",
        )
    )

    assert second.task_key == "req-2"
    assert CountingHandler.instances == 2


def test_handler_executor_uses_default_log_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from infly.runtime.log import ContextFilter, log_context
    import logging

    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    executor_logger = logging.getLogger("infly.runtime.executor")
    handler_loader_logger = logging.getLogger("infly.runtime.handler_loader")
    original_executor_level = executor_logger.level
    original_handler_loader_level = handler_loader_logger.level
    root.handlers = [caplog.handler]
    root.setLevel(logging.DEBUG)
    executor_logger.setLevel(logging.DEBUG)
    handler_loader_logger.setLevel(logging.INFO)
    caplog.handler.addFilter(ContextFilter())
    caplog.set_level(logging.DEBUG)
    try:
        registry = HandlerRegistry()
        registry.add(
            HandlerDefinition(
                handler_name="echo",
                entrypoint="tests.support.fake_handlers:EchoHandler",
            )
        )
        executor = HandlerExecutor(registry)

        with log_context():
            executor.preload()
            result = executor.execute(
                TaskRequest(
                    task_key="req-1",
                    handler_name="echo",
                    input={"text": "hello"},
                    caller="test",
                )
            )

        assert result.task_key == "req-1"
        assert any(
            record.message.startswith("handler_preload_started")
            and record.log_category == ""
            and record.log_name == "infly"
            for record in caplog.records
        )
        assert any(
            record.message.startswith("handler_load_started")
            and record.log_category == ""
            and record.log_name == "infly"
            for record in caplog.records
        )
        assert any(
            record.message.startswith("handler_execution_started")
            for record in caplog.records
        )
        assert any(
            record.message.startswith("handler_execution_completed")
            for record in caplog.records
        )
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)
        executor_logger.setLevel(original_executor_level)
        handler_loader_logger.setLevel(original_handler_loader_level)



