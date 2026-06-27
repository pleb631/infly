import threading
from concurrent.futures import Future

import pytest

from infly.core.contracts import TaskRequest, TaskResult, TaskStatus
from infly.core.errors import ErrorCode, PlatformError
from infly.core.handlers import HandlerDefinition
from infly.runtime.config import SchedulerConfig, WorkerGroup
from infly.runtime.registry import HandlerRegistry
from infly.runtime.scheduler import TaskScheduler
from infly.runtime.strategy.process_pool import ProcessPoolStrategy


def _registry(*definitions: HandlerDefinition) -> HandlerRegistry:
    registry = HandlerRegistry()
    for definition in definitions:
        registry.add(definition)
    return registry


def _request(
    task_key: str,
    handler_name: str = "echo",
) -> TaskRequest:
    return TaskRequest(
        task_key=task_key,
        handler_name=handler_name,
        input={"text": task_key},
        caller="integration",
    )


class ConcurrentBlockingStrategy:
    def __init__(self) -> None:
        self.release = threading.Event()
        self._lock = threading.Lock()
        self.running = 0
        self.max_running = 0
        self.first_started = threading.Event()
        self.both_started = threading.Event()

    def execute(self, request: TaskRequest) -> Future[TaskResult]:
        future: Future[TaskResult] = Future()

        with self._lock:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
            self.first_started.set()
            if self.running >= 2:
                self.both_started.set()

        def complete() -> None:
            self.release.wait()
            with self._lock:
                self.running -= 1
            future.set_result(
                TaskResult(
                    task_key=request.task_key,
                    output={"task_key": request.task_key},
                )
            )

        threading.Thread(target=complete, daemon=True).start()
        return future

    def close(self) -> None:
        self.release.set()


def test_submit_and_wait_runs_through_embedded_pool() -> None:
    pool = ProcessPoolStrategy(
        _registry(
            HandlerDefinition(
                handler_name="echo",
                entrypoint="tests.support.fake_handlers:ContextHandler",
            )
        ),
        [WorkerGroup(name="gpu", device="cuda:7")],
    )
    scheduler = TaskScheduler(
        pool,
        scheduler_config=SchedulerConfig(num_threads=1),
    )
    scheduler.start()
    try:
        result = scheduler.submit_and_wait(_request("hello"))
    finally:
        scheduler.stop()

    assert result.task_key == "hello"
    assert result.output["input"]["text"] == "hello"
    assert result.output["runtime_context"]["group_name"] == "gpu"
    assert result.output["runtime_context"]["device"] == "cuda:7"
    assert result.output["environment_device"] == "cuda:7"
    assert result.diagnostics["handler_name"] == "echo"
    assert result.diagnostics["caller"] == "integration"


def test_submit_and_wait_propagates_worker_failure() -> None:
    pool = ProcessPoolStrategy(
        _registry(
            HandlerDefinition(
                handler_name="broken",
                entrypoint="tests.support.fake_handlers:RaisingHandler",
            )
        ),
        [WorkerGroup(name="cpu", device="cpu")],
    )
    scheduler = TaskScheduler(
        pool,
        scheduler_config=SchedulerConfig(num_threads=1),
    )
    scheduler.start()
    try:
        with pytest.raises(PlatformError) as caught:
            scheduler.submit_and_wait(_request("boom", "broken"))
    finally:
        scheduler.stop()

    assert caught.value.code == ErrorCode.INTERNAL_ERROR
    assert "intentional prediction failure" in str(caught.value)


def test_scheduler_can_process_multiple_tasks_in_parallel() -> None:
    strategy = ConcurrentBlockingStrategy()
    scheduler = TaskScheduler(
        strategy,  # type: ignore[arg-type]
        scheduler_config=SchedulerConfig(num_threads=2),
    )
    scheduler.start()
    try:
        first = scheduler.submit(_request("first"))
        second = scheduler.submit(_request("second"))

        assert strategy.first_started.wait(1)
        assert strategy.both_started.wait(1)
        assert scheduler.query(first).status == TaskStatus.RUNNING
        assert scheduler.query(second).status == TaskStatus.RUNNING

        strategy.release.set()
        first_result = scheduler.query(first, wait=True)
        second_result = scheduler.query(second, wait=True)
    finally:
        scheduler.stop()

    assert strategy.max_running == 2
    assert first_result.status == TaskStatus.COMPLETED
    assert second_result.status == TaskStatus.COMPLETED

