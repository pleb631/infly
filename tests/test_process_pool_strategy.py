import logging
import threading
from collections import deque
from concurrent.futures import Future
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
    from infly.runtime.strategy.state import PoolLifecycleState

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
    pool._state = PoolLifecycleState()
    pool._state.add_workers([worker])
    pool._log_manager = FakeLogManager()
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
        request = _request("duplicate", "slow")
        original = pool.execute(request)
        duplicate = pool.execute(request)
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


def test_close_is_safe_when_called_concurrently() -> None:
    pool = ProcessPoolStrategy(
        _registry(_definition("echo")),
        [WorkerGroup(name="cpu")],
    )
    barrier = threading.Barrier(3)
    failures: list[BaseException] = []

    def close_pool() -> None:
        try:
            barrier.wait()
            pool.close()
        except BaseException as exc:  # pragma: no cover - assertion below records thread failures.
            failures.append(exc)

    first = threading.Thread(target=close_pool)
    second = threading.Thread(target=close_pool)
    first.start()
    second.start()
    barrier.wait()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert pool._state.close_complete is True


def test_concurrent_close_waits_for_resource_cleanup() -> None:
    """Every close caller must return only once shutdown is complete."""
    from types import SimpleNamespace

    from infly.runtime.strategy.state import PoolLifecycleState

    entered_stop = threading.Event()
    release_stop = threading.Event()
    second_returned = threading.Event()

    class BlockingController:
        def request_graceful_stop(self, worker) -> None:
            pass

        def stop(self, worker, *, terminate_after: float) -> None:
            entered_stop.set()
            assert release_stop.wait(timeout=2)

        def close_queues(self, workers) -> None:
            pass

    class FakeLogManager:
        def stop(self) -> None:
            pass

    pool = object.__new__(ProcessPoolStrategy)
    pool._state = PoolLifecycleState()
    pool._state.add_workers([SimpleNamespace(worker_id="cpu_R0", outstanding=0)])
    pool._worker_controller = BlockingController()
    pool._supervisor_stop = threading.Event()
    pool._result_stop = threading.Event()
    pool._log_manager = FakeLogManager()

    first = threading.Thread(target=pool.close)
    second = threading.Thread(target=lambda: (pool.close(), second_returned.set()))
    first.start()
    assert entered_stop.wait(timeout=1)
    second.start()
    assert not second_returned.wait(timeout=0.1)

    release_stop.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_returned.is_set()
    assert pool._state.close_complete is True


def test_close_waits_for_inflight_group_registration_cleanup() -> None:
    """Close must include workers that have started but are not yet published."""
    from infly.runtime.strategy.groups import WorkerGroupCatalog
    from infly.runtime.strategy.manager import WorkerManager
    from infly.runtime.strategy.router import ProcessPoolRouter
    from infly.runtime.strategy.state import PoolLifecycleState

    startup_waiting = threading.Event()
    dispose_started = threading.Event()
    release_dispose = threading.Event()
    close_returned = threading.Event()
    registration_failures: list[BaseException] = []

    class BlockingController:
        def launch(self, worker) -> None:
            worker.alive = True

        def await_ready(self, worker, deadline: float, *, stop_event) -> None:
            startup_waiting.set()
            assert stop_event.wait(timeout=2)
            raise PlatformError(ErrorCode.INTERNAL_ERROR, "startup interrupted")

        def dispose(self, workers) -> None:
            dispose_started.set()
            assert release_dispose.wait(timeout=2)

        def request_graceful_stop(self, worker) -> None:
            pass

        def stop(self, worker, *, terminate_after: float) -> None:
            pass

        def close_queues(self, workers) -> None:
            pass

    class FakeLogManager:
        def stop(self) -> None:
            pass

    state = PoolLifecycleState()
    catalog = WorkerGroupCatalog(HandlerRegistry())
    pool = object.__new__(ProcessPoolStrategy)
    pool._state = state
    pool._worker_controller = BlockingController()
    pool._supervisor_stop = threading.Event()
    pool._result_stop = threading.Event()
    pool._log_manager = FakeLogManager()
    manager = WorkerManager(
        state=state,
        catalog=catalog,
        router=ProcessPoolRouter(catalog.groups),
        controller=pool._worker_controller,
        startup_timeout_seconds=1,
        stop_event=pool._supervisor_stop,
        shutdown_pool=lambda _worker: None,
    )

    def register_group() -> None:
        try:
            manager.register(WorkerGroup(name="gpu"))
        except BaseException as exc:  # pragma: no cover - assertions inspect the recorded exception.
            registration_failures.append(exc)

    registration = threading.Thread(target=register_group)
    registration.start()
    assert startup_waiting.wait(timeout=1)

    closing = threading.Thread(target=lambda: (pool.close(), close_returned.set()))
    closing.start()
    assert dispose_started.wait(timeout=1)
    assert not close_returned.wait(timeout=0.1)

    release_dispose.set()
    registration.join(timeout=2)
    closing.join(timeout=2)

    assert not registration.is_alive()
    assert not closing.is_alive()
    assert len(registration_failures) == 1
    assert isinstance(registration_failures[0], PlatformError)
    assert close_returned.is_set()
    assert state.close_complete is True


def test_close_waits_for_inflight_group_unregistration_cleanup() -> None:
    """Close must not release shared queues while detached workers are stopping."""
    from infly.runtime.strategy.groups import WorkerGroupCatalog
    from infly.runtime.strategy.manager import WorkerManager
    from infly.runtime.strategy.router import ProcessPoolRouter
    from infly.runtime.strategy.state import PoolLifecycleState
    from infly.runtime.strategy.worker import WorkerState

    dispose_started = threading.Event()
    release_dispose = threading.Event()
    close_returned = threading.Event()

    class BlockingController:
        def dispose(self, workers) -> None:
            assert list(workers) == [worker]
            dispose_started.set()
            assert release_dispose.wait(timeout=2)

        def request_graceful_stop(self, worker) -> None:
            pass

        def stop(self, worker, *, terminate_after: float) -> None:
            pass

        def close_queues(self, workers) -> None:
            assert list(workers) == []

    class FakeLogManager:
        def stop(self) -> None:
            pass

    group = WorkerGroup(name="gpu")
    worker = WorkerState(worker_id="gpu_R0", index=0, group=group, handler_names=())
    state = PoolLifecycleState()
    state.add_workers([worker])
    catalog = WorkerGroupCatalog(HandlerRegistry())
    catalog.publish(group, ())
    controller = BlockingController()
    pool = object.__new__(ProcessPoolStrategy)
    pool._state = state
    pool._worker_controller = controller
    pool._supervisor_stop = threading.Event()
    pool._result_stop = threading.Event()
    pool._log_manager = FakeLogManager()
    manager = WorkerManager(
        state=state,
        catalog=catalog,
        router=ProcessPoolRouter(catalog.groups),
        controller=controller,
        startup_timeout_seconds=1,
        stop_event=pool._supervisor_stop,
        shutdown_pool=lambda _worker: None,
    )

    unloading = threading.Thread(target=lambda: manager.unregister("gpu"))
    unloading.start()
    assert dispose_started.wait(timeout=1)

    closing = threading.Thread(target=lambda: (pool.close(), close_returned.set()))
    closing.start()
    assert not close_returned.wait(timeout=0.1)

    release_dispose.set()
    unloading.join(timeout=2)
    closing.join(timeout=2)

    assert not unloading.is_alive()
    assert not closing.is_alive()
    assert close_returned.is_set()
    assert state.close_complete is True


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


def test_health_snapshot_treats_closed_process_as_not_alive() -> None:
    """Health checks must be safe while close releases process handles."""
    from infly.runtime.strategy.health import ProcessPoolHealthReporter
    from infly.runtime.strategy.state import PoolLifecycleState
    from infly.runtime.strategy.worker import WorkerState

    class ClosedProcess:
        def is_alive(self) -> bool:
            raise ValueError("process object is closed")

    group = WorkerGroup(name="cpu")
    worker = WorkerState(
        worker_id="cpu_R0",
        index=0,
        group=group,
        handler_names=(),
        process=ClosedProcess(),
        alive=True,
    )
    state = PoolLifecycleState()
    state.add_workers([worker])

    snapshot = ProcessPoolHealthReporter.snapshot(
        name="process_pool",
        groups={"cpu": group},
        state=state.snapshot(),
    )

    assert snapshot.status == HealthStatus.DEGRADED
    assert snapshot.detail["alive_workers"] == 0


def test_restart_launch_failure_is_cleaned_up_and_rescheduled() -> None:
    """A transient process-launch error must not kill the supervisor loop."""
    from threading import Event

    from infly.runtime.config import WorkerSafetyPolicy
    from infly.runtime.strategy.groups import WorkerGroupCatalog
    from infly.runtime.strategy.manager import WorkerManager
    from infly.runtime.strategy.router import ProcessPoolRouter
    from infly.runtime.strategy.state import PoolLifecycleState
    from infly.runtime.strategy.worker import WorkerState

    class FailingController:
        def __init__(self) -> None:
            self.cleaned_workers: list[str] = []

        def launch(self, worker: WorkerState) -> None:
            worker.generation += 1
            raise OSError("process creation temporarily unavailable")

        def cleanup_failed_start(self, worker: WorkerState) -> None:
            self.cleaned_workers.append(worker.worker_id)
            worker.alive = False
            worker.process = None

    group = WorkerGroup(
        name="cpu",
        safety=WorkerSafetyPolicy(
            mode="restart",
            restart_limit=2,
            restart_backoff_seconds=0,
        ),
    )
    state = PoolLifecycleState()
    worker = WorkerState(
        worker_id="cpu_R0",
        index=0,
        group=group,
        handler_names=(),
    )
    state.add_workers([worker])
    catalog = WorkerGroupCatalog(HandlerRegistry())
    catalog.publish(group, ())
    controller = FailingController()
    manager = WorkerManager(
        state=state,
        catalog=catalog,
        router=ProcessPoolRouter(catalog.groups),
        controller=controller,
        startup_timeout_seconds=1,
        stop_event=Event(),
        shutdown_pool=lambda _worker: None,
    )

    manager._restart(worker)

    assert controller.cleaned_workers == ["cpu_R0"]
    assert worker.next_restart_at is not None
    assert len(worker.restart_times) == 1


def test_restart_does_not_hold_pool_state_lock_while_launching() -> None:
    """A slow process launch must not pause unrelated request scheduling."""
    from threading import Event

    from infly.runtime.config import WorkerSafetyPolicy
    from infly.runtime.strategy.groups import WorkerGroupCatalog
    from infly.runtime.strategy.manager import WorkerManager
    from infly.runtime.strategy.router import ProcessPoolRouter
    from infly.runtime.strategy.state import PoolLifecycleState
    from infly.runtime.strategy.worker import WorkerState

    launch_started = Event()
    release_launch = Event()

    class BlockingController:
        def launch(self, worker: WorkerState) -> None:
            launch_started.set()
            assert release_launch.wait(timeout=2)
            worker.generation += 1

        def await_ready(self, worker: WorkerState, deadline: float, stop_event: Event) -> None:
            worker.alive = True

        def cleanup_failed_start(self, worker: WorkerState) -> None:
            pass

    group = WorkerGroup(name="cpu", safety=WorkerSafetyPolicy(mode="restart", restart_backoff_seconds=0))
    worker = WorkerState(worker_id="cpu_R0", index=0, group=group, handler_names=(), alive=True)
    state = PoolLifecycleState()
    state.add_workers([worker])
    catalog = WorkerGroupCatalog(HandlerRegistry())
    catalog.publish(group, ())
    manager = WorkerManager(
        state=state,
        catalog=catalog,
        router=ProcessPoolRouter(catalog.groups),
        controller=BlockingController(),
        startup_timeout_seconds=1,
        stop_event=Event(),
        shutdown_pool=lambda _worker: None,
    )

    restart = threading.Thread(target=manager._restart, args=(worker,))
    restart.start()
    assert launch_started.wait(timeout=1)
    assert state.lock.acquire(blocking=False)
    state.lock.release()

    # Dispatch holds the pool-state lock while selecting workers.  Lifecycle
    # work must not make that selection wait for a slow process launch.
    routing_finished = Event()
    routed_workers: list[dict[str, list[WorkerState]]] = []

    def select_workers() -> None:
        with state.lock:
            routed_workers.append(manager._router.live_workers_by_group([worker]))
        routing_finished.set()

    routing = threading.Thread(target=select_workers)
    routing.start()
    try:
        assert routing_finished.wait(timeout=0.1)
    finally:
        release_launch.set()
        routing.join(timeout=2)
        restart.join(timeout=2)

    assert not routing.is_alive()
    assert not restart.is_alive()
    assert routed_workers == [{}]
    assert worker.alive is True


def test_await_ready_reports_stopped_worker_without_attribute_error() -> None:
    """A concurrent close may clear the process handle during startup."""
    from time import monotonic

    from infly.runtime.strategy.lifecycle import WorkerProcessController
    from infly.runtime.strategy.worker import WorkerState

    worker = WorkerState(
        worker_id="cpu_R0",
        index=0,
        group=WorkerGroup(name="cpu"),
        handler_names=(),
        lifecycle_queue=None,
        process=None,
    )
    controller = object.__new__(WorkerProcessController)

    with pytest.raises(PlatformError, match="stopped during startup"):
        controller.await_ready(worker, monotonic() + 1)


def test_shutdown_after_worker_failure_completes_resource_cleanup() -> None:
    """The shutdown safety policy must leave the pool in its terminal state."""
    from types import SimpleNamespace

    from infly.runtime.strategy.state import PoolLifecycleState

    class FakeController:
        def __init__(self) -> None:
            self.stopped: list[str] = []
            self.closed_workers: list[str] = []

        def stop(self, worker, *, terminate_after: float) -> None:
            self.stopped.append(worker.worker_id)

        def close_queues(self, workers) -> None:
            self.closed_workers = [worker.worker_id for worker in workers]

    class FakeLogManager:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    failed_worker = SimpleNamespace(worker_id="cpu_R0", generation=1, outstanding=0)
    remaining_worker = SimpleNamespace(worker_id="cpu_R1", generation=1, outstanding=0)
    pool = object.__new__(ProcessPoolStrategy)
    pool._state = PoolLifecycleState()
    pool._state.add_workers([failed_worker, remaining_worker])
    pending: Future[TaskResult] = Future()
    pool._state.register_task("pending", pending)
    pool._worker_controller = FakeController()
    pool._supervisor_stop = threading.Event()
    pool._result_stop = threading.Event()
    pool._log_manager = FakeLogManager()

    pool._shutdown_after_worker_failure(failed_worker)

    with pytest.raises(PlatformError) as caught:
        pending.result()
    assert caught.value.code == ErrorCode.WORKER_UNAVAILABLE
    assert pool._state.close_complete is True
    assert pool._supervisor_stop.is_set()
    assert pool._result_stop.is_set()
    assert pool._worker_controller.stopped == ["cpu_R0", "cpu_R1"]
    assert pool._worker_controller.closed_workers == ["cpu_R0", "cpu_R1"]
    assert pool._log_manager.stopped is True


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
