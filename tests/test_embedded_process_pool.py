import logging
from collections import deque
from dataclasses import dataclass

import pytest

from infly.core.contracts import InferenceRequest, InferenceResult
from infly.core.errors import ErrorCode, PlatformError

from infly.core.models import ModelDefinition
from infly.runtime.log import ContextFilter, LoggingSettings
from infly.runtime.config import WorkerGroup
from infly.runtime.registry import ModelRegistry
from infly.runtime.strategy.embedded_process_pool import (
    EmbeddedProcessPoolStrategy,
    _worker_loop,
)


def _registry(*definitions: ModelDefinition) -> ModelRegistry:
    registry = ModelRegistry()
    for definition in definitions:
        registry.add(definition)
    return registry


def _definition(
    name: str,
    class_name: str = "ContextModel",
    **kwargs: object,
) -> ModelDefinition:
    return ModelDefinition(
        model_name=name,
        class_path=f"tests.support.fake_models:{class_name}",
        kwargs=kwargs,
    )


def _request(request_id: str, model_name: str = "echo") -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        model_name=model_name,
        payload={"text": request_id},
        caller="test",
    )


def test_pool_validates_groups_and_deployed_models() -> None:
    registry = _registry(_definition("echo"))

    with pytest.raises(PlatformError) as caught:
        EmbeddedProcessPoolStrategy(registry, [])
    assert caught.value.code == ErrorCode.INTERNAL_ERROR

    duplicate_groups = [
        WorkerGroup(name="same", device="cpu"),
        WorkerGroup(name="same", device="cpu"),
    ]
    with pytest.raises(PlatformError, match="unique"):
        EmbeddedProcessPoolStrategy(registry, duplicate_groups)

    with pytest.raises(PlatformError, match="missing"):
        EmbeddedProcessPoolStrategy(
            registry,
            [WorkerGroup(name="cpu", device="cpu", models=["missing"])],
        )


def test_pool_injects_distinct_worker_context_without_mutating_registry() -> None:
    definition = _definition("echo")
    registry = _registry(definition)
    pool = EmbeddedProcessPoolStrategy(
        registry,
        [
            WorkerGroup(
                name="gpu",
                device="cuda:7",
                process_count=2,
                environment={"INFLY_TEST_ENV": "configured"},
            )
        ],
    )
    try:
        results = [
            pool.execute(_request(f"request-{index}")).result(timeout=3)
            for index in range(4)
        ]
    finally:
        pool.close()

    contexts = [result.data["worker_context"] for result in results]
    assert {context["group_name"] for context in contexts} == {"gpu"}
    assert {context["device"] for context in contexts} == {"cuda:7"}
    assert {context["worker_id"] for context in contexts} == {"gpu_R0", "gpu_R1"}
    assert {result.data["environment_device"] for result in results} == {"cuda:7"}
    assert {result.data["custom_environment"] for result in results} == {
        "configured"
    }
    assert definition.module_dict == {}


def test_pool_only_routes_models_deployed_to_a_group() -> None:
    registry = _registry(_definition("deployed"), _definition("idle"))
    pool = EmbeddedProcessPoolStrategy(
        registry,
        [WorkerGroup(name="cpu", device="cpu", models=["deployed"])],
    )
    try:
        result = pool.execute(_request("ok", "deployed")).result(timeout=3)
        unavailable = pool.execute(_request("missing", "idle"))
        with pytest.raises(PlatformError) as caught:
            unavailable.result(timeout=1)
    finally:
        pool.close()

    assert result.request_id == "ok"
    assert caught.value.code == ErrorCode.WORKER_UNAVAILABLE


def test_pool_fails_construction_when_model_preload_fails() -> None:
    registry = _registry(_definition("broken", "FailingModel"))

    with pytest.raises(PlatformError) as caught:
        EmbeddedProcessPoolStrategy(
            registry,
            [WorkerGroup(name="cpu", device="cpu")],
            startup_timeout_seconds=2,
        )

    assert caught.value.code == ErrorCode.INTERNAL_ERROR
    assert "startup" in str(caught.value).lower()


def test_abort_startup_closes_worker_and_result_queues(monkeypatch) -> None:
    from types import SimpleNamespace

    import infly.runtime.strategy.embedded_process_pool as pool_module

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

    pool = object.__new__(EmbeddedProcessPoolStrategy)
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

    pool_module.EmbeddedProcessPoolStrategy._abort_startup(pool)

    assert worker.task_queue is None
    assert worker.lifecycle_queue is None
    assert pool._result_queue.closed is True
    assert pool._result_queue.joined is True
    assert worker.process.closed is True
    assert pool._log_manager.stopped is True


def test_pool_startup_timeout_is_internal_error() -> None:
    registry = _registry(
        _definition("slow", "SlowInitModel", delay_seconds=1)
    )

    with pytest.raises(PlatformError) as caught:
        EmbeddedProcessPoolStrategy(
            registry,
            [WorkerGroup(name="cpu", device="cpu")],
            startup_timeout_seconds=0.05,
        )

    assert caught.value.code == ErrorCode.INTERNAL_ERROR
    assert "timed out" in str(caught.value).lower()


def test_empty_model_list_preloads_all_registry_models() -> None:
    registry = _registry(
        _definition("healthy"),
        _definition("broken", "FailingModel"),
    )

    with pytest.raises(PlatformError, match="startup"):
        EmbeddedProcessPoolStrategy(
            registry,
            [WorkerGroup(name="all", device="cpu", models=[])],
            startup_timeout_seconds=2,
        )

    pool = EmbeddedProcessPoolStrategy(
        registry,
        [WorkerGroup(name="selected", device="cpu", models=["healthy"])],
    )
    pool.close()


def test_cross_group_routing_is_weighted_by_live_process_count() -> None:
    pool = EmbeddedProcessPoolStrategy(
        _registry(_definition("echo")),
        [
            WorkerGroup(name="small", device="cpu", process_count=1),
            WorkerGroup(name="large", device="cpu", process_count=2),
        ],
    )
    try:
        results = [
            pool.execute(_request(f"weighted-{index}")).result(timeout=3)
            for index in range(6)
        ]
    finally:
        pool.close()

    group_names = [
        result.data["worker_context"]["group_name"] for result in results
    ]
    assert group_names.count("small") == 2
    assert group_names.count("large") == 4


def test_duplicate_request_and_model_failure_are_internal_errors() -> None:
    pool = EmbeddedProcessPoolStrategy(
        _registry(
            _definition("slow", "SlowModel", delay_seconds=0.2),
            _definition("broken", "RaisingPredictModel"),
        ),
        [WorkerGroup(name="cpu", device="cpu")],
    )
    try:
        original = pool.execute(_request("duplicate", "slow"))
        duplicate = pool.execute(_request("duplicate", "slow"))
        with pytest.raises(PlatformError) as duplicate_error:
            duplicate.result(timeout=1)
        with pytest.raises(PlatformError) as model_error:
            pool.execute(_request("broken", "broken")).result(timeout=3)
        original.result(timeout=3)
    finally:
        pool.close()

    assert duplicate_error.value.code == ErrorCode.INTERNAL_ERROR
    assert model_error.value.code == ErrorCode.INTERNAL_ERROR


def test_close_is_idempotent_and_fails_pending_future() -> None:
    pool = EmbeddedProcessPoolStrategy(
        _registry(_definition("slow", "SlowModel", delay_seconds=5)),
        [WorkerGroup(name="cpu", device="cpu", models=["slow"])],
    )
    pending = pool.execute(_request("pending", "slow"))

    pool.close()
    pool.close()

    with pytest.raises(PlatformError) as caught:
        pending.result(timeout=1)
    assert caught.value.code == ErrorCode.INTERNAL_ERROR


def test_close_stops_logging_listener() -> None:
    pool = EmbeddedProcessPoolStrategy(
        _registry(_definition("echo")),
        [WorkerGroup(name="cpu", device="cpu")],
    )

    pool.close()

    assert not pool.log_manager.listener.thread.is_alive()


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

    class FakeService:
        def __init__(self, registry: ModelRegistry) -> None:
            self.registry = registry

        def preload(self) -> None:
            logging.getLogger("fake.service").info("service_preload_called")

        def predict(self, request: InferenceRequest) -> InferenceResult:
            logging.getLogger("fake.service").info(
                "service_predict_called request_id=%s",
                request.request_id,
            )
            return InferenceResult(
                request_id=request.request_id,
                data={"result": "ok"},
            )

    import infly.runtime.strategy.embedded_process_pool as pool_module

    caplog.handler.addFilter(ContextFilter())
    caplog.set_level(logging.INFO)

    monkeypatch.setattr(pool_module, "setup_worker_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(pool_module, "setproctitle", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pool_module,
        "_restore_parent_import_path",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(pool_module, "InferenceService", FakeService)

    task_queue = FakeQueue(
        [
            InferenceRequest(
                request_id="req-1",
                model_name="echo",
                payload={"text": "hello"},
                caller="test",
            ),
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
        registry=ModelRegistry(),
        environment={},
        device="cpu",
        parent_sys_path=[],
        parent_cwd="",
        log_queue=None,
        log_settings=LoggingSettings(),
    )

    assert any(
        record.message == "service_preload_called"
        and record.log_category == "worker"
        and record.log_name == "worker-1"
        for record in caplog.records
    )
    assert any(
        record.message.startswith("service_predict_called")
        and record.log_category == "worker"
        and record.log_name == "worker-1"
        for record in caplog.records
    )
    assert lifecycle_queue.put_items[0].kind == "READY"
    assert result_queue.put_items[0].ok is True
    assert result_queue.put_items[0].payload.request_id == "req-1"


