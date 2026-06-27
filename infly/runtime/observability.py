from __future__ import annotations

import datetime
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from statistics import fmean
from typing import Any

from infly.core.contracts import TaskRequest
from infly.core.errors import ErrorCode


class HealthStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


@dataclass(slots=True, frozen=True)
class StrategyHealthSnapshot:
    name: str
    status: HealthStatus
    accepting: bool
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SchedulerHealthSnapshot:
    status: HealthStatus
    accepting: bool
    started: bool
    closed: bool
    worker_threads: int
    outstanding_tasks: int
    max_outstanding_tasks: int
    backend_status_counts: Mapping[str, int]
    strategy: StrategyHealthSnapshot | None = None


@dataclass(slots=True, frozen=True)
class RuntimeMetricsSnapshot:
    submitted_total: int
    started_total: int
    completed_total: int
    failed_total: int
    inflight_tasks: int
    latency_count: int
    last_latency_ms: float | None
    latency_ms_avg: float | None
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    latency_ms_max: float | None


@dataclass(slots=True, frozen=True)
class TraceEvent:
    name: str
    timestamp: datetime.datetime
    task_id: str
    task_key: str
    handler_name: str
    caller: str
    trace_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None
    error_code: str | None = None
    error_message: str | None = None


TraceSink = Callable[[TraceEvent], None]


def _percentile(sorted_values: list[float], quantile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = round((len(sorted_values) - 1) * quantile)
    return sorted_values[index]


class RuntimeInstrumentation:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._submitted_total = 0
        self._started_total = 0
        self._completed_total = 0
        self._failed_total = 0
        self._inflight_tasks = 0
        self._submitted_at: dict[str, float] = {}
        self._latency_ms: list[float] = []
        self._last_latency_ms: float | None = None
        self._trace_sinks: list[TraceSink] = []

    def add_trace_sink(self, sink: TraceSink) -> None:
        with self._lock:
            self._trace_sinks.append(sink)

    def remove_trace_sink(self, sink: TraceSink) -> None:
        with self._lock:
            self._trace_sinks = [candidate for candidate in self._trace_sinks if candidate is not sink]

    def record_submitted(self, task_id: str, request: TaskRequest) -> None:
        with self._lock:
            self._submitted_total += 1
            self._inflight_tasks += 1
            self._submitted_at[task_id] = time.perf_counter()
        self._emit(
            TraceEvent(
                name="task.submitted",
                timestamp=datetime.datetime.now(datetime.UTC),
                task_id=task_id,
                task_key=request.task_key,
                handler_name=request.handler_name,
                caller=request.caller,
                trace_id=_trace_id(request),
                metadata=dict(request.metadata),
            )
        )

    def record_started(self, task_id: str, request: TaskRequest) -> None:
        with self._lock:
            self._started_total += 1
        self._emit(
            TraceEvent(
                name="task.started",
                timestamp=datetime.datetime.now(datetime.UTC),
                task_id=task_id,
                task_key=request.task_key,
                handler_name=request.handler_name,
                caller=request.caller,
                trace_id=_trace_id(request),
                metadata=dict(request.metadata),
            )
        )

    def record_completed(self, task_id: str, request: TaskRequest) -> None:
        duration_ms = self._finish(task_id, completed=True)
        self._emit(
            TraceEvent(
                name="task.completed",
                timestamp=datetime.datetime.now(datetime.UTC),
                task_id=task_id,
                task_key=request.task_key,
                handler_name=request.handler_name,
                caller=request.caller,
                trace_id=_trace_id(request),
                metadata=dict(request.metadata),
                duration_ms=duration_ms,
            )
        )

    def record_failed(
        self,
        task_id: str,
        request: TaskRequest,
        *,
        error_code: ErrorCode | str | None,
        error_message: str | None,
    ) -> None:
        duration_ms = self._finish(task_id, completed=False)
        code = error_code.value if isinstance(error_code, ErrorCode) else error_code
        self._emit(
            TraceEvent(
                name="task.failed",
                timestamp=datetime.datetime.now(datetime.UTC),
                task_id=task_id,
                task_key=request.task_key,
                handler_name=request.handler_name,
                caller=request.caller,
                trace_id=_trace_id(request),
                metadata=dict(request.metadata),
                duration_ms=duration_ms,
                error_code=code,
                error_message=error_message,
            )
        )

    def _finish(self, task_id: str, *, completed: bool) -> float | None:
        now = time.perf_counter()
        with self._lock:
            started_at = self._submitted_at.pop(task_id, None)
            if self._inflight_tasks > 0:
                self._inflight_tasks -= 1
            if completed:
                self._completed_total += 1
            else:
                self._failed_total += 1
            if started_at is None:
                return None
            duration_ms = (now - started_at) * 1000
            self._last_latency_ms = duration_ms
            self._latency_ms.append(duration_ms)
            return duration_ms

    def metrics_snapshot(self) -> RuntimeMetricsSnapshot:
        with self._lock:
            latencies = sorted(self._latency_ms)
            avg = fmean(latencies) if latencies else None
            return RuntimeMetricsSnapshot(
                submitted_total=self._submitted_total,
                started_total=self._started_total,
                completed_total=self._completed_total,
                failed_total=self._failed_total,
                inflight_tasks=self._inflight_tasks,
                latency_count=len(latencies),
                last_latency_ms=self._last_latency_ms,
                latency_ms_avg=avg,
                latency_ms_p50=_percentile(latencies, 0.50),
                latency_ms_p95=_percentile(latencies, 0.95),
                latency_ms_max=latencies[-1] if latencies else None,
            )

    def render_prometheus_text(self) -> str:
        metrics = self.metrics_snapshot()
        lines = [
            "# HELP infly_tasks_submitted_total Total submitted tasks.",
            "# TYPE infly_tasks_submitted_total counter",
            f"infly_tasks_submitted_total {metrics.submitted_total}",
            "# HELP infly_tasks_started_total Total started tasks.",
            "# TYPE infly_tasks_started_total counter",
            f"infly_tasks_started_total {metrics.started_total}",
            "# HELP infly_tasks_completed_total Total completed tasks.",
            "# TYPE infly_tasks_completed_total counter",
            f"infly_tasks_completed_total {metrics.completed_total}",
            "# HELP infly_tasks_failed_total Total failed tasks.",
            "# TYPE infly_tasks_failed_total counter",
            f"infly_tasks_failed_total {metrics.failed_total}",
            "# HELP infly_tasks_inflight Current inflight tasks.",
            "# TYPE infly_tasks_inflight gauge",
            f"infly_tasks_inflight {metrics.inflight_tasks}",
        ]
        if metrics.last_latency_ms is not None:
            lines.extend(
                [
                    "# HELP infly_task_latency_last_milliseconds Last observed task latency.",
                    "# TYPE infly_task_latency_last_milliseconds gauge",
                    f"infly_task_latency_last_milliseconds {metrics.last_latency_ms}",
                ]
            )
        if metrics.latency_ms_p50 is not None:
            lines.extend(
                [
                    "# HELP infly_task_latency_p50_milliseconds Median task latency.",
                    "# TYPE infly_task_latency_p50_milliseconds gauge",
                    f"infly_task_latency_p50_milliseconds {metrics.latency_ms_p50}",
                ]
            )
        if metrics.latency_ms_p95 is not None:
            lines.extend(
                [
                    "# HELP infly_task_latency_p95_milliseconds P95 task latency.",
                    "# TYPE infly_task_latency_p95_milliseconds gauge",
                    f"infly_task_latency_p95_milliseconds {metrics.latency_ms_p95}",
                ]
            )
        return "\n".join(lines) + "\n"

    def _emit(self, event: TraceEvent) -> None:
        with self._lock:
            sinks = list(self._trace_sinks)
        for sink in sinks:
            sink(event)


def _trace_id(request: TaskRequest) -> str | None:
    trace_id = request.metadata.get("trace_id")
    if trace_id is None:
        return None
    return str(trace_id)


__all__ = [
    "HealthStatus",
    "RuntimeInstrumentation",
    "RuntimeMetricsSnapshot",
    "SchedulerHealthSnapshot",
    "StrategyHealthSnapshot",
    "TraceEvent",
]
