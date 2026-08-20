import logging
from collections import deque
from dataclasses import dataclass

import pytest

from infly.core.contracts import TaskRequest, TaskResult
from infly.core.errors import ErrorCode, PlatformError
from infly.core.handlers import HandlerDefinition
from infly.runtime.config import WorkerGroup
from infly.runtime.log import ContextFilter, LoggingSettings
from infly.runtime.observability import HealthStatus
from infly.runtime.registry import HandlerRegistry
from infly.runtime.strategy.process_pool import (
    ProcessPoolStrategy,
    _worker_loop,
)


def _registry(*definitions: HandlerDefinition) -> HandlerRegistry:
    registry = HandlerRegistry()
    for definition in definitions:
        registry.add(definition)
    return registry


def _definition(
    name: str,
    class_name: str = "ContextHandler",
    **kwargs: object,
) -> HandlerDefinition:
    return HandlerDefinition(
        handler_name=name,
        entrypoint=f"tests.support.fake_handlers:{class_name}",
        init_kwargs=kwargs,
    )


def _request(task_id: str, handler_name: str = "echo") -> TaskRequest:
    return TaskRequest(
        handler_name=handler_name,
        input={"text": task_id},
    )


def test_pool_validates_groups_and_deployed_handlers() -> None:
    registry = _registry(_definition("echo"))

    with pytest.raises(PlatformError) as caught:
        ProcessPoolStrategy(registry, [])
    assert caught.value.code == ErrorCode.INTERNAL_ERROR

    duplicate_groups = [
        WorkerGroup(name="same"),
        WorkerGroup(name="same"),
    ]
    with pytest.raises(PlatformError, match="unique"):
        ProcessPoolStrategy(registry, duplicate_groups)

    with pytest.raises(PlatformError, match="missing"):
        ProcessPoolStrategy(
            registry,
            [WorkerGroup(name="cpu", handlers=["missing"])],
        )


def test_pool_injects_distinct_worker_context_without_mutating_registry() -> None:
    definition = _definition("echo")
    registry = _registry(definition)
    pool = ProcessPoolStrategy(
        registry,
        [
            WorkerGroup(
                name="gpu",
                process_count=2,
                environment={"INFLY_TEST_ENV": "configured"},
            )
        ],
    )
    try:
        results = [pool.execute(_request(f"request-{index}")).result(timeout=3) for index in range(4)]
    finally:
        pool.close()

    contexts = [result.output["runtime_context"] for result in results]
    assert {context["group_name"] for context in contexts} == {"gpu"}
    assert {context["worker_id"] for context in contexts} == {"gpu_R0", "gpu_R1"}
    assert {result.output["custom_environment"] for result in results} == {"configured"}
    assert definition.init_context == {}


def test_pool_only_routes_handlers_deployed_to_a_group() -> None:
    registry = _registry(_definition("deployed"), _definition("idle"))
    pool = ProcessPoolStrategy(
        registry,
        [WorkerGroup(name="cpu", handlers=["deployed"])],
    )
    try:
        request = _request("ok", "deployed")
        result = pool.execute(request).result(timeout=3)
        unavailable = pool.execute(_request("missing", "idle"))
        with pytest.raises(PlatformError) as caught:
            unavailable.result(timeout=1)
    finally:
        pool.close()

    assert result.task_id == request.task_id
    assert caught.value.code == ErrorCode.WORKER_UNAVAILABLE


def test_worker_groups_can_be_registered_and_unloaded_at_runtime() -> None:
    pool = ProcessPoolStrategy(
        _registry(_definition("cpu"), _definition("gpu")),
        [WorkerGroup(name="cpu", handlers=["cpu"])],
    )
    try:
        pool.register_worker_group(
            WorkerGroup(name="gpu", handlers=["gpu"]),
        )

        result = pool.execute(_request("gpu-request", "gpu")).result(timeout=3)
        snapshot = pool.health_snapshot()
        pool.unregister_worker_group("gpu")

        unavailable = pool.execute(_request("after-unload", "gpu"))
        with pytest.raises(PlatformError) as caught:
            unavailable.result(timeout=1)
        unloaded_snapshot = pool.health_snapshot()
    finally:
        pool.close()

    assert result.output["runtime_context"]["group_name"] == "gpu"
    assert snapshot.detail["groups"]["gpu"]["alive_workers"] == 1
    assert caught.value.code == ErrorCode.WORKER_UNAVAILABLE
    assert "gpu" not in unloaded_snapshot.detail["groups"]


def test_unloading_group_fails_its_assigned_requests() -> None:
    pool = ProcessPoolStrategy(
        _registry(_definition("slow", "SlowHandler", delay_seconds=5)),
        [WorkerGroup(name="cpu", handlers=["slow"])],
    )
    try:
        pending = pool.execute(_request("pending-unload", "slow"))
        pool.unload_worker_group("cpu")
        with pytest.raises(PlatformError) as caught:
            pending.result(timeout=1)
        snapshot = pool.health_snapshot()
    finally:
        pool.close()

    assert caught.value.code == ErrorCode.WORKER_UNAVAILABLE
    assert snapshot.status == HealthStatus.DOWN
    assert snapshot.detail["groups"] == {}


def test_runtime_worker_group_registration_validates_name_and_handlers() -> None:
    pool = ProcessPoolStrategy(
        _registry(_definition("echo")),
        [WorkerGroup(name="cpu")],
    )
    try:
        with pytest.raises(PlatformError, match="already registered"):
            pool.register_worker_group(WorkerGroup(name="cpu"))
        with pytest.raises(PlatformError, match="missing"):
            pool.register_worker_group(
                WorkerGroup(name="missing", handlers=["unknown"]),
            )
        with pytest.raises(PlatformError) as caught:
            pool.unregister_worker_group("unknown")
    finally:
        pool.close()

    assert caught.value.code == ErrorCode.NOT_FOUND


def test_failed_runtime_worker_group_registration_is_not_published() -> None:
    pool = ProcessPoolStrategy(
        _registry(_definition("healthy"), _definition("broken", "FailingHandler")),
        [WorkerGroup(name="cpu", handlers=["healthy"])],
        startup_timeout_seconds=2,
    )
    try:
        with pytest.raises(PlatformError, match="startup failed"):
            pool.register_worker_group(
                WorkerGroup(name="broken", handlers=["broken"]),
            )
        result = pool.execute(_request("still-healthy", "healthy")).result(timeout=3)
        snapshot = pool.health_snapshot()
    finally:
        pool.close()

    assert result.output["runtime_context"]["group_name"] == "cpu"
    assert set(snapshot.detail["groups"]) == {"cpu"}


def test_pool_fails_construction_when_handler_preload_fails() -> None:
    registry = _registry(_definition("broken", "FailingHandler"))

    with pytest.raises(PlatformError) as caught:
        ProcessPoolStrategy(
            registry,
            [WorkerGroup(name="cpu")],
            startup_timeout_seconds=2,
        )

    assert caught.value.code == ErrorCode.INTERNAL_ERROR
    assert "startup" in str(caught.value).lower()


def test_abort_startup_closes_worker_and_result_queues(monkeypatch) -> None:
    from types import SimpleNamespace

    import infly.runtime.strategy.process_pool as pool_module

    class FakeQueue:
        def __init__(self) -> None:
            self.closed = False
            self.joined = False

        def close(self) -> None:
            self.closed = True

        def join_thread(self) -> None:
            self.joined = True

    class FakeProcess:
        def __init__(self) -> None:
            self.joined: list[float] = []
            self.closed = False

        def is_alive(self) -> bool:
            return False

        def join(self, timeout=None) -> None:
            self.joined.append(timeout)

        def close(self) -> None:
            self.closed = True

    class FakeLogManager:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    pool = object.__new__(ProcessPoolStrategy)
    worker = SimpleNamespace(
        worker_id="cpu_R0",
        generation=1,
        process=FakeProcess(),
        task_queue=FakeQueue(),
        lifecycle_queue=FakeQueue(),
        alive=True,
    )
    pool._workers = {"cpu_R0": worker}
    pool._log_manager = FakeLogManager()
    pool._accepting = True
    pool._closing = False
    pool._result_queue = FakeQueue()

    pool_module.ProcessPoolStrategy._abort_startup(pool)

    assert worker.task_queue is None
    assert worker.lifecycle_queue is None
    assert pool._result_queue.closed is True
    assert pool._result_queue.joined is True
    assert worker.process.closed is True
    assert pool._log_manager.stopped is True


def test_pool_startup_timeout_is_internal_error() -> None:
    registry = _registry(_definition("slow", "SlowInitHandler", delay_seconds=1))

    with pytest.raises(PlatformError) as caught:
        ProcessPoolStrategy(
            registry,
            [WorkerGroup(name="cpu")],
            startup_timeout_seconds=0.05,
        )

    assert caught.value.code == ErrorCode.INTERNAL_ERROR
    assert "timed out" in str(caught.value).lower()


def test_empty_handler_list_preloads_all_registry_handlers() -> None:
    registry = _registry(
        _definition("healthy"),
        _definition("broken", "FailingHandler"),
    )

    with pytest.raises(PlatformError, match="startup"):
        ProcessPoolStrategy(
            registry,
            [WorkerGroup(name="all", handlers=[])],
            startup_timeout_seconds=2,
        )

    pool = ProcessPoolStrategy(
        registry,
        [WorkerGroup(name="selected", handlers=["healthy"])],
    )
    pool.close()


def test_cross_group_routing_is_weighted_by_live_process_count() -> None:
    pool = ProcessPoolStrategy(
        _registry(_definition("echo")),
        [
            WorkerGroup(name="small", process_count=1),
            WorkerGroup(name="large", process_count=2),
        ],
    )
    try:
        results = [pool.execute(_request(f"weighted-{index}")).result(timeout=3) for index in range(6)]
    finally:
        pool.close()

    group_names = [result.output["runtime_context"]["group_name"] for result in results]
    assert group_names.count("small") == 2
    assert group_names.count("large") == 4


def test_duplicate_request_and_handler_failure_are_internal_errors() -> None:
    pool = ProcessPoolStrategy(
        _registry(
            _definition("slow", "SlowHandler", delay_seconds=0.2),
            _definition("broken", "RaisingHandler"),
        ),
        [WorkerGroup(name="cpu")],
    )
    try:
        original = pool.execute(_request("duplicate", "slow"))
        duplicate = pool.execute(_request("duplicate", "slow"))
        with pytest.raises(PlatformError) as duplicate_error:
            duplicate.result(timeout=1)
        with pytest.raises(PlatformError) as handler_error:
            pool.execute(_request("broken", "broken")).result(timeout=3)
        original.result(timeout=3)
    finally:
        pool.close()

    assert duplicate_error.value.code == ErrorCode.INTERNAL_ERROR
    assert handler_error.value.code == ErrorCode.INTERNAL_ERROR


def test_close_is_idempotent_and_fails_pending_future() -> None:
    pool = ProcessPoolStrategy(
        _registry(_definition("slow", "SlowHandler", delay_seconds=5)),
        [WorkerGroup(name="cpu", handlers=["slow"])],
    )
    pending = pool.execute(_request("pending", "slow"))

    pool.close()
    pool.close()

    with pytest.raises(PlatformError) as caught:
        pending.result(timeout=1)
    assert caught.value.code == ErrorCode.INTERNAL_ERROR


def test_close_stops_logging_listener() -> None:
    pool = ProcessPoolStrategy(
        _registry(_definition("echo")),
        [WorkerGroup(name="cpu")],
    )

    pool.close()

    assert not pool.log_manager.listener.thread.is_alive()


def test_health_snapshot_reports_live_workers_and_groups() -> None:
    pool = ProcessPoolStrategy(
        _registry(_definition("echo")),
        [
            WorkerGroup(name="cpu", process_count=2),
            WorkerGroup(name="gpu", process_count=1),
        ],
    )
    try:
        snapshot = pool.health_snapshot()
    finally:
        pool.close()

    assert snapshot.name == "process_pool"
    assert snapshot.status == HealthStatus.OK
    assert snapshot.accepting is True
    assert snapshot.detail["total_workers"] == 3
    assert snapshot.detail["alive_workers"] == 3
    assert snapshot.detail["groups"]["cpu"]["configured_processes"] == 2
    assert snapshot.detail["groups"]["cpu"]["alive_workers"] == 2
    assert snapshot.detail["groups"]["gpu"]["configured_processes"] == 1
    assert snapshot.detail["groups"]["gpu"]["alive_workers"] == 1


def test_worker_loop_applies_log_context_in_worker_layer(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    @dataclass
    class FakeQueue:
        def __init__(self, items: list[object] | None = None) -> None:
            self.items = deque(items or [])
            self.put_items: list[object] = []

        def get(self):
            return self.items.popleft()

        def put(self, item, timeout=None):
            self.put_items.append(item)

    class FakeExecutor:
        def __init__(self, registry: HandlerRegistry) -> None:
            self.registry = registry

        def preload(self) -> None:
            logging.getLogger("fake.executor").info("executor_preload_called")

        def execute(self, request: TaskRequest) -> TaskResult:
            logging.getLogger("fake.executor").info(
                "executor_execute_called task_id=%s",
                request.task_id,
            )
            return TaskResult(
                task_id=request.task_id,
                output={"result": "ok"},
            )

    import infly.runtime.strategy.process_pool as pool_module

    caplog.handler.addFilter(ContextFilter())
    caplog.set_level(logging.INFO)

    monkeypatch.setattr(pool_module, "setup_worker_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(pool_module, "setproctitle", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pool_module,
        "_restore_parent_import_path",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(pool_module, "HandlerExecutor", FakeExecutor)

    request = TaskRequest(
        handler_name="echo",
        input={"text": "hello"},
    )
    task_queue = FakeQueue(
        [
            request,
            None,
        ]
    )
    result_queue = FakeQueue()
    lifecycle_queue = FakeQueue()

    _worker_loop(
        worker_id="worker-1",
        generation=1,
        task_queue=task_queue,
        result_queue=result_queue,
        lifecycle_queue=lifecycle_queue,
        registry=HandlerRegistry(),
        environment={},
        parent_sys_path=[],
        parent_cwd="",
        log_queue=None,
        log_settings=LoggingSettings(),
    )

    assert any(
        record.message == "executor_preload_called"
        and record.log_category == "worker"
        and record.log_name == "worker-1"
        for record in caplog.records
    )
    assert any(
        record.message.startswith("executor_execute_called")
        and record.log_category == "worker"
        and record.log_name == "worker-1"
        for record in caplog.records
    )
    assert lifecycle_queue.put_items[0].kind == "READY"
    assert result_queue.put_items[0].ok is True
    assert result_queue.put_items[0].payload.task_id == request.task_id
