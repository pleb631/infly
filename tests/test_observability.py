import time
from concurrent.futures import Future

from infly.core.contracts import TaskRequest, TaskResult
from infly.core.errors import ErrorCode, PlatformError
from infly.runtime.config import SchedulerConfig
from infly.runtime.observability import (
    HealthStatus,
    RuntimeInstrumentation,
    StrategyHealthSnapshot,
)
from infly.runtime.scheduler import TaskScheduler


def _request(task_key: str, handler_name: str = "echo") -> TaskRequest:
    return TaskRequest(
        task_key=task_key,
        handler_name=handler_name,
        input={"text": task_key},
        caller="test",
        metadata={"trace_id": f"trace-{task_key}"},
    )


def _wait_for_terminal(scheduler: TaskScheduler, task_id: str) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        response = scheduler.query(task_id)
        if response.status.value in {"COMPLETED", "FAILED"}:
            return
        time.sleep(0.01)
    scheduler.query(task_id, wait=True)


class SuccessThenFailStrategy:
    def execute(self, request: TaskRequest) -> Future[TaskResult]:
        future: Future[TaskResult] = Future()
        if request.handler_name == "broken":
            future.set_exception(
                PlatformError(ErrorCode.WORKER_UNAVAILABLE, "worker unavailable")
            )
        else:
            future.set_result(
                TaskResult(
                    task_key=request.task_key,
                    output={"echo": request.input["text"]},
                )
            )
        return future

    def close(self) -> None:
        pass

    def health_snapshot(self) -> StrategyHealthSnapshot:
        return StrategyHealthSnapshot(
            name="fake",
            status=HealthStatus.OK,
            accepting=True,
            detail={"kind": "test-double"},
        )


def test_scheduler_health_snapshot_reports_runtime_state() -> None:
    scheduler = TaskScheduler(
        SuccessThenFailStrategy(),
        instrumentation=RuntimeInstrumentation(),
        scheduler_config=SchedulerConfig(num_workers=1),
    )
    scheduler.start()
    try:
        task_id = scheduler.submit(_request("health"))
        _wait_for_terminal(scheduler, task_id)
        snapshot = scheduler.health_snapshot()
    finally:
        scheduler.stop()

    assert snapshot.status == HealthStatus.OK
    assert snapshot.accepting is True
    assert snapshot.worker_threads == 1
    assert snapshot.outstanding_tasks == 0
    assert snapshot.backend_status_counts["COMPLETED"] == 1
    assert snapshot.strategy is not None
    assert snapshot.strategy.name == "fake"
    assert snapshot.strategy.detail == {"kind": "test-double"}


def test_runtime_instrumentation_emits_trace_events_and_metrics() -> None:
    instrumentation = RuntimeInstrumentation()
    trace_events = []
    instrumentation.add_trace_sink(trace_events.append)

    scheduler = TaskScheduler(
        SuccessThenFailStrategy(),
        instrumentation=instrumentation,
        scheduler_config=SchedulerConfig(num_workers=1),
    )
    scheduler.start()
    try:
        first_task_id = scheduler.submit(_request("first"))
        second_task_id = scheduler.submit(_request("second", handler_name="broken"))
        _wait_for_terminal(scheduler, first_task_id)
        _wait_for_terminal(scheduler, second_task_id)
    finally:
        scheduler.stop()

    metrics = instrumentation.metrics_snapshot()
    prometheus_text = instrumentation.render_prometheus_text()

    assert metrics.submitted_total == 2
    assert metrics.started_total == 2
    assert metrics.completed_total == 1
    assert metrics.failed_total == 1
    assert metrics.inflight_tasks == 0
    assert metrics.last_latency_ms is not None
    assert metrics.latency_count == 2

    assert "infly_tasks_submitted_total 2" in prometheus_text
    assert "infly_tasks_completed_total 1" in prometheus_text
    assert "infly_tasks_failed_total 1" in prometheus_text

    event_names = [event.name for event in trace_events]
    assert event_names == [
        "task.submitted",
        "task.submitted",
        "task.started",
        "task.completed",
        "task.started",
        "task.failed",
    ]
    assert trace_events[0].task_id == first_task_id
    assert trace_events[0].trace_id == "trace-first"
    assert trace_events[3].duration_ms is not None
    assert trace_events[5].task_id == second_task_id
    assert trace_events[5].error_code == ErrorCode.WORKER_UNAVAILABLE.value
