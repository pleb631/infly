from __future__ import annotations

from dataclasses import asdict

from infly import (
    HandlerDefinition,
    HandlerRegistry,
    PlatformError,
    ProcessPoolStrategy,
    RuntimeInstrumentation,
    SchedulerConfig,
    TaskRequest,
    TaskScheduler,
    TraceEvent,
    WorkerGroup,
)


def _print_trace(event: TraceEvent) -> None:
    duration = "n/a" if event.duration_ms is None else f"{event.duration_ms:.3f}ms"
    print(
        f"{event.name} task_id={event.task_id} task_key={event.task_key} "
        f"handler={event.handler_name} trace_id={event.trace_id} "
        f"duration={duration} error_code={event.error_code}"
    )


def main() -> int:
    trace_events: list[TraceEvent] = []
    instrumentation = RuntimeInstrumentation()
    instrumentation.add_trace_sink(trace_events.append)

    registry = HandlerRegistry()
    registry.add(
        HandlerDefinition(
            handler_name="echo",
            entrypoint="infly.demo.handlers:build_demo_echo_handler",
            init_kwargs={"prefix": "[demo] "},
        )
    )
    registry.add(
        HandlerDefinition(
            handler_name="broken",
            entrypoint="infly.demo.handlers:build_demo_unavailable_handler",
            init_kwargs={"message": "demo worker unavailable"},
        )
    )

    worker_groups = [
        WorkerGroup(
            name="cpu",
            device="cpu",
            process_count=1,
            handlers=["echo", "broken"],
            environment={"INFLY_DEMO": "1"},
        )
    ]
    strategy = ProcessPoolStrategy(registry, worker_groups)
    scheduler = TaskScheduler(
        strategy,
        instrumentation=instrumentation,
        scheduler_config=SchedulerConfig(
            max_outstanding_tasks=8,
            num_threads=1,
            max_retained_terminal_tasks=8,
        ),
    )

    print("=== SETUP ===")
    print("handlers=['echo', 'broken']")
    print("worker_group=cpu device=cpu process_count=1")
    print()

    scheduler.start()
    try:
        success_request = TaskRequest(
            task_key="demo-success",
            handler_name="echo",
            input={"text": "hello quickstart"},
            caller="quickstart",
            metadata={"trace_id": "trace-demo-success"},
        )
        async_request = TaskRequest(
            task_key="demo-async",
            handler_name="echo",
            input={"text": "async hello"},
            caller="quickstart",
            metadata={"trace_id": "trace-demo-async"},
        )
        failed_request = TaskRequest(
            task_key="demo-failure",
            handler_name="broken",
            input={"text": "boom"},
            caller="quickstart",
            metadata={"trace_id": "trace-demo-failure"},
        )

        result = scheduler.submit_and_wait(success_request)
        print("=== SUCCESS RESULT ===")
        print(asdict(result))
        print()

        async_task_id = scheduler.submit(async_request)
        query_response = scheduler.query(async_task_id, wait=True)
        print("=== QUERY RESULT ===")
        print(asdict(query_response))
        print()

        print("=== FAILURE RESULT ===")
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
