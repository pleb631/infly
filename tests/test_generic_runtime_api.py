import pytest

from infly.core.contracts import TaskRequest
from infly.core.errors import ErrorCode, PlatformError
from infly.core.handlers import HandlerDefinition
from infly.runtime.config import WorkerGroup
from infly.runtime.executor import HandlerExecutor
from infly.runtime.registry import HandlerRegistry


def test_handler_registry_get_missing_raises_platform_error() -> None:
    registry = HandlerRegistry()

    with pytest.raises(PlatformError) as exc:
        registry.get("missing")

    assert exc.value.code == ErrorCode.HANDLER_NOT_FOUND
    assert "missing" in str(exc.value)


def test_handler_definition_rejects_reserved_runtime_context() -> None:
    with pytest.raises(ValueError, match="runtime_context"):
        HandlerDefinition(
            handler_name="echo",
            entrypoint="tests.support.fake_handlers:EchoHandler",
            init_context={"runtime_context": {}},
        )


def test_handler_definition_injects_runtime_context_without_mutation() -> None:
    definition = HandlerDefinition(
        handler_name="echo",
        entrypoint="tests.support.fake_handlers:EchoHandler",
        init_context={"gpu": 0},
    )

    runtime_definition = HandlerDefinition.with_runtime_context(
        definition,
        runtime_context={"worker_id": "cpu_R0"},
    )

    assert runtime_definition.init_context["runtime_context"] == {"worker_id": "cpu_R0"}
    assert runtime_definition.init_context["gpu"] == 0
    assert definition.init_context == {"gpu": 0}


def test_handler_executor_executes_registered_handler() -> None:
    registry = HandlerRegistry()
    registry.add(
        HandlerDefinition(
            handler_name="echo",
            entrypoint="tests.support.fake_handlers:EchoHandler",
            init_context={"prefix": "ok:"},
        )
    )
    executor = HandlerExecutor(registry)

    result = executor.execute(
        TaskRequest(
            task_key="task-1",
            handler_name="echo",
            input={"text": "hello"},
            caller="test",
        )
    )

    assert result.task_key == "task-1"
    assert result.output == {"echo": "ok:hello"}
    assert result.diagnostics["handler_name"] == "echo"
    assert result.diagnostics["caller"] == "test"


def test_handler_executor_recreates_transient_handler_instances() -> None:
    from tests.support.fake_handlers import CountingHandler

    CountingHandler.instances = 0
    registry = HandlerRegistry()
    registry.add(
        HandlerDefinition(
            handler_name="echo",
            entrypoint="tests.support.fake_handlers:CountingHandler",
            reuse_mode="task_transient",
        )
    )
    executor = HandlerExecutor(registry)

    first = executor.execute(
        TaskRequest(
            task_key="task-1",
            handler_name="echo",
            input={"text": "one"},
            caller="test",
        )
    )
    second = executor.execute(
        TaskRequest(
            task_key="task-2",
            handler_name="echo",
            input={"text": "two"},
            caller="test",
        )
    )

    assert first.task_key == "task-1"
    assert second.task_key == "task-2"
    assert CountingHandler.instances == 2


def test_handler_executor_preload_skips_transient_handler_instances() -> None:
    from tests.support.fake_handlers import CountingHandler

    CountingHandler.instances = 0
    registry = HandlerRegistry()
    registry.add(
        HandlerDefinition(
            handler_name="echo",
            entrypoint="tests.support.fake_handlers:CountingHandler",
            reuse_mode="task_transient",
        )
    )
    executor = HandlerExecutor(registry)

    executor.preload()

    assert CountingHandler.instances == 0


def test_handler_executor_keeps_worker_cached_handler_instances() -> None:
    from tests.support.fake_handlers import CountingHandler

    CountingHandler.instances = 0
    registry = HandlerRegistry()
    registry.add(
        HandlerDefinition(
            handler_name="echo",
            entrypoint="tests.support.fake_handlers:CountingHandler",
        )
    )
    executor = HandlerExecutor(registry)

    first = executor.execute(
        TaskRequest(
            task_key="task-1",
            handler_name="echo",
            input={"text": "one"},
            caller="test",
        )
    )
    second = executor.execute(
        TaskRequest(
            task_key="task-2",
            handler_name="echo",
            input={"text": "two"},
            caller="test",
        )
    )

    assert first.task_key == "task-1"
    assert second.task_key == "task-2"
    assert CountingHandler.instances == 1


def test_worker_group_accepts_handlers_and_rejects_duplicates() -> None:
    group = WorkerGroup(
        name="cpu",
        device="cpu",
        handlers=["echo", "other"],
    )

    assert group.handlers == ["echo", "other"]

    with pytest.raises(ValueError, match="duplicates"):
        WorkerGroup(name="cpu", device="cpu", handlers=["echo", "echo"])

