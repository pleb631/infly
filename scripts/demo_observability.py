from __future__ import annotations

from concurrent.futures import Future
from dataclasses import asdict

from infly import (
    ErrorCode,
    PlatformError,
    RuntimeInstrumentation,
    SchedulerConfig,
    TaskRequest,
    TaskResult,
    TaskScheduler,
)
from infly.runtime.observability import HealthStatus, StrategyHealthSnapshot, TraceEvent


class DemoStrategy:
    def execute(self, request: TaskRequest) -> Future[TaskResult]:
        future: Future[TaskResult] = Future()
        if request.handler_name == "broken":
            future.set_exception(
                PlatformError(ErrorCode.WORKER_UNAVAILABLE, "demo worker unavailable")
            )
        else:
            future.set_result(
                TaskResult(
                    task_key=request.task_key,
                    output={
                        "echo": request.input["text"],
                        "trace_id": request.metadata.get("trace_id"),
                    },
                    diagnostics={"mode": "demo"},
                )
            )
        return future

    def close(self) -> None:
        pass

    def health_snapshot(self) -> StrategyHealthSnapshot:
        return StrategyHealthSnapshot(
            name="demo_strategy",
            status=HealthStatus.OK,
            accepting=True,
            detail={"kind": "inline-demo"},
        )


def _print_trace(event: TraceEvent) -> None:
    duration = "n/a" if event.duration_ms is None else f"{event.duration_ms:.3f}ms"
    print(
        f"{event.name} task_id={event.task_id} task_key={event.task_key} "
        f"trace_id={event.trace_id} duration={duration} error_code={event.error_code}"
    )


def main() -> int:
    instrumentation = RuntimeInstrumentation()
    trace_events: list[TraceEvent] = []
    instrumentation.add_trace_sink(trace_events.append)

    scheduler = TaskScheduler(
        DemoStrategy(),
        instrumentation=instrumentation,
        scheduler_config=SchedulerConfig(num_workers=1),
    )

    success_request = TaskRequest(
        task_key="demo-success",
        handler_name="echo",
        input={"text": "hello observability"},
        caller="demo",
        metadata={"trace_id": "trace-demo-success"},
    )
    failed_request = TaskRequest(
        task_key="demo-failure",
        handler_name="broken",
        input={"text": "boom"},
        caller="demo",
        metadata={"trace_id": "trace-demo-failure"},
    )

    scheduler.start()
    try:
        result = scheduler.submit_and_wait(success_request)
        print("=== TASK RESULT ===")
        print(asdict(result))
        print()

        print("=== TASK FAILURE ===")
        try:
            scheduler.submit_and_wait(failed_request)
        except PlatformError as exc:
            print(f"code={exc.code.value} message={exc}")
        print()

        print("=== HEALTH ===")
        print(asdict(scheduler.health_snapshot()))
        print()

        metrics = instrumentation.metrics_snapshot()
        print("=== METRICS ===")
        print(
            " ".join(
                [
                    f"submitted_total={metrics.submitted_total}",
                    f"started_total={metrics.started_total}",
                    f"completed_total={metrics.completed_total}",
                    f"failed_total={metrics.failed_total}",
                    f"inflight_tasks={metrics.inflight_tasks}",
                ]
            )
        )
        print(asdict(metrics))
        print()

        print("=== PROMETHEUS ===")
        print(instrumentation.render_prometheus_text().rstrip())
        print()

        print("=== TRACES ===")
        for event in trace_events:
            _print_trace(event)
    finally:
        scheduler.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
