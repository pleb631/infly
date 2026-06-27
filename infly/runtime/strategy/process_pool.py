from __future__ import annotations

import os
import sys
import threading
import time
from collections import defaultdict, deque
from collections.abc import Mapping
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass, field
from multiprocessing import Queue, get_context
from queue import Empty
from typing import Any

from setproctitle import setproctitle

from infly.core.contracts import TaskRequest, TaskResult
from infly.core.errors import ErrorCode, PlatformError
from infly.core.handlers import HandlerDefinition
from infly.runtime.config import WorkerGroup
from infly.runtime.executor import HandlerExecutor
from infly.runtime.log import (
    LoggingSettings,
    MainLogManager,
    get_logger,
    log_context,
    setup_main_logging,
    setup_worker_logging,
)
from infly.runtime.observability import HealthStatus, StrategyHealthSnapshot
from infly.runtime.registry import HandlerRegistry


def _dump_error_code(code: object) -> object:
    return code.value if isinstance(code, ErrorCode) else code


def _load_error_code(code: object) -> ErrorCode:
    if isinstance(code, ErrorCode):
        return code
    try:
        return ErrorCode(code)
    except Exception:
        return ErrorCode.INTERNAL_ERROR


def _restore_parent_import_path(
    parent_sys_path: list[str],
    parent_cwd: str,
) -> None:
    missing: list[str] = []
    for entry in parent_sys_path:
        absolute_entry = parent_cwd if entry == "" else entry
        if absolute_entry not in sys.path:
            missing.append(absolute_entry)
    if missing:
        sys.path = missing + sys.path


@dataclass(slots=True, frozen=True)
class WorkerLifecycleMessage:
    kind: str
    worker_id: str
    generation: int
    error_message: str | None = None


@dataclass(slots=True, frozen=True)
class WorkerResultMessage:
    ok: bool
    task_key: str
    worker_id: str
    generation: int
    payload: TaskResult | None = None
    error_code: ErrorCode | object | None = None
    error_message: str | None = None


def _worker_loop(
    worker_id: str,
    generation: int,
    task_queue: Queue,
    result_queue: Queue,
    lifecycle_queue: Queue,
    registry: HandlerRegistry,
    environment: Mapping[str, str],
    device: str,
    parent_sys_path: list[str],
    parent_cwd: str,
    log_queue: Queue,
    log_settings: LoggingSettings,
) -> None:

    setup_worker_logging(log_queue, settings=log_settings)
    setproctitle(f"INFLY::{worker_id}")

    log = get_logger(name=worker_id, category="worker")
    log.info(
        "worker_started worker_id=%s generation=%s device=%s",
        worker_id,
        generation,
        device,
    )
    try:
        _restore_parent_import_path(parent_sys_path, parent_cwd)
        os.environ.update(environment)
        os.environ["INFLY_DEVICE"] = device
        executor = HandlerExecutor(registry)
        with log_context(name=worker_id, category="worker"):
            executor.preload()
        lifecycle_queue.put(
            WorkerLifecycleMessage(
                kind="READY",
                worker_id=worker_id,
                generation=generation,
            )
        )
        log.info(
            "worker_ready worker_id=%s generation=%s",
            worker_id,
            generation,
        )
    except Exception as exc:
        log.error(
            "worker_startup_failed worker_id=%s generation=%s error=%s",
            worker_id,
            generation,
            exc,
            exc_info=True,
        )
        lifecycle_queue.put(
            WorkerLifecycleMessage(
                kind="STARTUP_FAILED",
                worker_id=worker_id,
                generation=generation,
                error_message=str(exc),
            )
        )
        return

    while True:
        request = task_queue.get()
        if request is None:
            log.info(
                "worker_stopped worker_id=%s generation=%s",
                worker_id,
                generation,
            )
            return
        try:
            log.debug(
                "worker_request_started worker_id=%s generation=%s task_key=%s handler=%s",
                worker_id,
                generation,
                request.task_key,
                request.handler_name,
            )
            with log_context(name=worker_id, category="worker"):
                result = executor.execute(request)
            result_queue.put(
                WorkerResultMessage(
                    ok=True,
                    payload=result,
                    task_key=request.task_key,
                    worker_id=worker_id,
                    generation=generation,
                )
            )
            log.debug(
                "worker_request_completed worker_id=%s generation=%s task_key=%s",
                worker_id,
                generation,
                request.task_key,
            )
        except Exception as exc:
            log.error(
                "worker_request_failed worker_id=%s generation=%s task_key=%s error=%s",
                worker_id,
                generation,
                request.task_key,
                exc,
                exc_info=True,
            )
            result_queue.put(
                WorkerResultMessage(
                    ok=False,
                    error_code=_dump_error_code(getattr(exc, "code", ErrorCode.INTERNAL_ERROR)),
                    error_message=str(exc),
                    task_key=request.task_key,
                    worker_id=worker_id,
                    generation=generation,
                )
            )


@dataclass(slots=True)
class _WorkerState:
    worker_id: str
    index: int
    group: WorkerGroup
    handler_names: tuple[str, ...]
    generation: int = 0
    task_queue: Any = None
    lifecycle_queue: Any = None
    process: Any = None
    alive: bool = False
    outstanding: int = 0
    restart_times: deque[float] = field(default_factory=deque)
    next_restart_at: float | None = None


log = get_logger("infly")


class ProcessPoolStrategy:
    name = "process_pool"

    def __init__(
        self,
        registry: HandlerRegistry,
        worker_groups: list[WorkerGroup],
        *,
        startup_timeout_seconds: float = 300,
    ) -> None:

        self._registry = registry
        self._startup_timeout_seconds = startup_timeout_seconds
        self._groups: dict[str, WorkerGroup] = {}
        self._group_handlers: dict[str, tuple[str, ...]] = {}
        self._handler_groups: dict[str, list[str]] = defaultdict(list)
        self._workers: dict[str, _WorkerState] = {}
        self._group_cursors: dict[str, int] = defaultdict(int)
        self._smooth_weights: dict[str, int] = defaultdict(int)
        self._futures: dict[str, Future[TaskResult]] = {}
        self._assignments: dict[str, tuple[str, int]] = {}
        self._lock = threading.RLock()
        self._mp_context = get_context("spawn")
        self._result_stop = threading.Event()
        self._supervisor_stop = threading.Event()
        self._accepting = True
        self._closing = False
        self._close_complete = False
        if os.name == "nt":
            self._queue_factory = Queue
        else:
            self._queue_factory = self._mp_context.Queue
        try:
            self._log_manager: MainLogManager = setup_main_logging(
                mp_context=self._mp_context,
                start=True,
            )
            self._log_settings = self._log_manager.settings
            log.info(
                "pool_starting groups=%s startup_timeout=%s",
                len(worker_groups),
                startup_timeout_seconds,
            )
            if not worker_groups:
                raise PlatformError(
                    ErrorCode.INTERNAL_ERROR,
                    "ProcessPoolStrategy requires at least one worker group",
                )
            if startup_timeout_seconds <= 0:
                raise PlatformError(
                    ErrorCode.INTERNAL_ERROR,
                    "startup_timeout_seconds must be greater than zero",
                )

            group_names = [group.name for group in worker_groups]
            if len(group_names) != len(set(group_names)):
                raise PlatformError(
                    ErrorCode.INTERNAL_ERROR,
                    "WorkerGroup names must be unique",
                )

            self._groups = {group.name: group for group in worker_groups}
            all_handler_names = tuple(definition.handler_name for definition in registry.list())
            for group in worker_groups:
                handler_names = tuple(group.handlers) if group.handlers else all_handler_names
                for handler_name in handler_names:
                    try:
                        registry.get(handler_name)
                    except PlatformError as exc:
                        raise PlatformError(
                            ErrorCode.INTERNAL_ERROR,
                            f"WorkerGroup '{group.name}' references missing handler "
                            f"'{handler_name}'",
                        ) from exc
                    self._handler_groups[handler_name].append(group.name)
                self._group_handlers[group.name] = handler_names
                for index in range(group.process_count):
                    worker_id = f"{group.name}_R{index}"
                    self._workers[worker_id] = _WorkerState(
                        worker_id=worker_id,
                        index=index,
                        group=group,
                        handler_names=handler_names,
                    )

            self._result_queue: Queue = self._queue_factory()
            for worker in self._workers.values():
                self._launch_worker(worker)
            self._await_initial_startup()
        except Exception as exc:
            self._abort_startup()
            if isinstance(exc, PlatformError):
                raise
            raise PlatformError(
                ErrorCode.INTERNAL_ERROR,
                f"Worker pool startup failed: {exc}",
            ) from exc

        self._result_thread = threading.Thread(
            target=self._result_loop,
            name="EmbeddedProcessPoolResultLoop",
            daemon=True,
        )
        self._supervisor_thread = threading.Thread(
            target=self._supervisor_loop,
            name="EmbeddedProcessPoolSupervisor",
            daemon=True,
        )
        self._result_thread.start()
        self._supervisor_thread.start()
        log.info("pool_started workers=%s", len(self._workers))

    def execute(
        self,
        request: TaskRequest,
    ) -> Future[TaskResult]:
        future: Future[TaskResult] = Future()
        with self._lock:
            if not self._accepting:
                log.warning(
                    "request_rejected task_key=%s reason=pool_closed",
                    request.task_key,
                )
                future.set_exception(
                    PlatformError(
                        ErrorCode.INTERNAL_ERROR,
                        "ProcessPoolStrategy is closed",
                    )
                )
                return future

            if request.task_key in self._futures:
                log.warning(
                    "request_rejected task_key=%s reason=duplicate",
                    request.task_key,
                )
                future.set_exception(
                    PlatformError(
                        ErrorCode.INTERNAL_ERROR,
                        f"Duplicate task_key: {request.task_key}",
                    )
                )
                return future

            group_names = [
                group_name
                for group_name in self._handler_groups.get(request.handler_name, [])
                if self._alive_workers_locked(group_name)
            ]
            if not group_names:
                log.warning(
                    "request_rejected task_key=%s handler=%s reason=no_live_worker",
                    request.task_key,
                    request.handler_name,
                )
                future.set_exception(
                    PlatformError(
                        ErrorCode.WORKER_UNAVAILABLE,
                        f"No live worker is deployed for handler '{request.handler_name}'",
                    )
                )
                return future

            self._futures[request.task_key] = future
            remaining_groups = list(group_names)
            while remaining_groups:
                group_name = self._select_group_locked(remaining_groups)
                for worker in self._ordered_workers_locked(group_name):
                    assignment = (worker.worker_id, worker.generation)
                    self._assignments[request.task_key] = assignment
                    worker.outstanding += 1
                    try:
                        worker.task_queue.put_nowait(request)
                    except Exception as exc:
                        worker.outstanding -= 1
                        self._assignments.pop(request.task_key, None)
                        log.warning(
                            "request_assignment_failed task_key=%s worker_id=%s "
                            "generation=%s error=%s",
                            request.task_key,
                            worker.worker_id,
                            worker.generation,
                            exc,
                            exc_info=True,
                        )
                        continue

                    self._group_cursors[group_name] = (
                        worker.index + 1
                    ) % worker.group.process_count
                    log.debug(
                        "request_assigned task_key=%s handler=%s worker_id=%s generation=%s",
                        request.task_key,
                        request.handler_name,
                        worker.worker_id,
                        worker.generation,
                    )
                    return future
                remaining_groups.remove(group_name)

            self._futures.pop(request.task_key, None)
            log.warning(
                "request_rejected task_key=%s handler=%s reason=assignment_failed",
                request.task_key,
                request.handler_name,
            )
            future.set_exception(
                PlatformError(
                    ErrorCode.WORKER_UNAVAILABLE,
                    f"Unable to submit request to a live worker for handler "
                    f"'{request.handler_name}'",
                )
            )
            return future

    def close(self) -> None:
        with self._lock:
            if self._close_complete:
                log.debug("pool_close_skipped reason=already_closed")
                return
            self._accepting = False
            self._closing = True

        log.info("pool_closing workers=%s pending=%s", len(self._workers), len(self._futures))
        self._supervisor_stop.set()
        for worker in self._workers.values():
            if worker.process is None or not worker.process.is_alive():
                continue
            with suppress(Exception):
                worker.task_queue.put(None, timeout=0.2)

        for worker in self._workers.values():
            self._stop_process(worker, terminate_after=1)

        self._result_stop.set()
        if hasattr(self, "_result_thread"):
            self._result_thread.join(timeout=2)
        if (
            hasattr(self, "_supervisor_thread")
            and threading.current_thread() is not self._supervisor_thread
        ):
            self._supervisor_thread.join(timeout=2)

        self._fail_all_pending(
            PlatformError(
                ErrorCode.INTERNAL_ERROR,
                "ProcessPoolStrategy closed before request completed",
            )
        )
        self._close_queues()
        self._log_manager.stop()
        with self._lock:
            self._close_complete = True
        log.info("pool_closed")

    @property
    def log_manager(self) -> MainLogManager:
        return self._log_manager

    def health_snapshot(self) -> StrategyHealthSnapshot:
        with self._lock:
            total_workers = len(self._workers)
            alive_workers = sum(
                1
                for worker in self._workers.values()
                if worker.alive and worker.process is not None and worker.process.is_alive()
            )
            restarting_workers = sum(
                1 for worker in self._workers.values() if worker.next_restart_at is not None
            )
            degraded_workers = total_workers - alive_workers - restarting_workers

            if self._close_complete or (not self._accepting and alive_workers == 0):
                status = HealthStatus.DOWN
            elif alive_workers == total_workers:
                status = HealthStatus.OK
            else:
                status = HealthStatus.DEGRADED

            groups: dict[str, dict[str, int | bool]] = {}
            for group_name, group in self._groups.items():
                group_workers = [
                    worker for worker in self._workers.values() if worker.group.name == group_name
                ]
                group_alive = sum(
                    1
                    for worker in group_workers
                    if worker.alive and worker.process is not None and worker.process.is_alive()
                )
                groups[group_name] = {
                    "configured_processes": group.process_count,
                    "alive_workers": group_alive,
                    "accepting": self._accepting,
                }

        return StrategyHealthSnapshot(
            name=self.name,
            status=status,
            accepting=self._accepting,
            detail={
                "total_workers": total_workers,
                "alive_workers": alive_workers,
                "restarting_workers": restarting_workers,
                "degraded_workers": max(degraded_workers, 0),
                "groups": groups,
            },
        )

    def _launch_worker(self, worker: _WorkerState) -> None:
        worker.generation += 1
        worker.alive = False
        worker.outstanding = 0
        worker.next_restart_at = None
        worker.task_queue = self._queue_factory()
        worker.lifecycle_queue = self._queue_factory(1)
        child_registry = self._worker_registry(worker)
        log.info(
            "worker_launching worker_id=%s generation=%s",
            worker.worker_id,
            worker.generation,
        )
        worker.process = self._mp_context.Process(
            target=_worker_loop,
            args=(
                worker.worker_id,
                worker.generation,
                worker.task_queue,
                self._result_queue,
                worker.lifecycle_queue,
                child_registry,
                worker.group.environment,
                worker.group.device,
                list(sys.path),
                os.getcwd(),
                self._log_manager.queue,
                self._log_settings,
            ),
            daemon=True,
            name=worker.worker_id,
        )
        worker.process.start()
        log.info(
            "worker_launched worker_id=%s generation=%s pid=%s",
            worker.worker_id,
            worker.generation,
            worker.process.pid,
        )

    def _worker_registry(self, worker: _WorkerState) -> HandlerRegistry:
        child_registry = HandlerRegistry()
        context = {
            "group_name": worker.group.name,
            "worker_id": worker.worker_id,
            "device": worker.group.device,
        }
        for handler_name in worker.handler_names:
            definition = self._registry.get(handler_name)
            child_registry.add(
                HandlerDefinition.with_runtime_context(
                    definition,
                    runtime_context=context,
                )
            )
        return child_registry

    def _await_initial_startup(self) -> None:
        deadline = time.monotonic() + self._startup_timeout_seconds
        for worker in self._workers.values():
            self._await_worker_ready(worker, deadline)

    def _await_worker_ready(
        self,
        worker: _WorkerState,
        deadline: float,
        stop_event: threading.Event | None = None,
    ) -> None:
        while True:
            if stop_event is not None and stop_event.is_set():
                raise PlatformError(
                    ErrorCode.INTERNAL_ERROR,
                    f"Worker '{worker.worker_id}' startup was interrupted",
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PlatformError(
                    ErrorCode.INTERNAL_ERROR,
                    f"Worker pool startup timed out waiting for '{worker.worker_id}'",
                )
            try:
                message = worker.lifecycle_queue.get(timeout=min(remaining, 0.1))
                break
            except Empty:
                if not worker.process.is_alive():
                    raise PlatformError(
                        ErrorCode.INTERNAL_ERROR,
                        f"Worker '{worker.worker_id}' exited during startup",
                    ) from Empty

        if message.kind != "READY" or message.generation != worker.generation:
            log.error(
                "worker_startup_failed worker_id=%s generation=%s error=%s",
                worker.worker_id,
                worker.generation,
                message.error_message or "unknown error",
            )
            raise PlatformError(
                ErrorCode.INTERNAL_ERROR,
                f"Worker '{worker.worker_id}' startup failed: "
                f"{message.error_message or 'unknown error'}",
            )
        worker.alive = True
        log.info(
            "worker_ready worker_id=%s generation=%s",
            worker.worker_id,
            worker.generation,
        )

    def _abort_startup(self) -> None:
        log.error("pool_startup_aborted")
        self._accepting = False
        self._closing = True
        for worker in self._workers.values():
            self._stop_process(worker, terminate_after=0)
        self._close_queues()
        manager = getattr(self, "_log_manager", None)
        if manager is not None:
            manager.stop()

    def _close_queue(self, queue: object | None) -> None:
        if queue is None:
            return
        close = getattr(queue, "close", None)
        if close is not None:
            with suppress(Exception):
                close()
        join_thread = getattr(queue, "join_thread", None)
        if join_thread is not None:
            with suppress(Exception):
                join_thread()

    def _close_queues(self) -> None:
        for worker in self._workers.values():
            self._close_queue(getattr(worker, "task_queue", None))
            self._close_queue(getattr(worker, "lifecycle_queue", None))
            worker.task_queue = None
            worker.lifecycle_queue = None
        self._close_queue(getattr(self, "_result_queue", None))

    def _stop_process(
        self,
        worker: _WorkerState,
        *,
        terminate_after: float,
    ) -> None:
        process = worker.process
        if process is None:
            return
        process.join(timeout=terminate_after)
        if process.is_alive():
            log.warning(
                "worker_terminating worker_id=%s generation=%s pid=%s",
                worker.worker_id,
                worker.generation,
                process.pid,
            )
            process.terminate()
            process.join(timeout=1)
        close = getattr(process, "close", None)
        if close is not None:
            with suppress(Exception):
                close()
        worker.alive = False
        log.info(
            "worker_stopped worker_id=%s generation=%s",
            worker.worker_id,
            worker.generation,
        )

    def _alive_workers_locked(self, group_name: str) -> list[_WorkerState]:
        return [
            worker
            for worker in self._workers.values()
            if worker.group.name == group_name
            and worker.alive
            and worker.process is not None
            and worker.process.is_alive()
        ]

    def _select_group_locked(self, group_names: list[str]) -> str:
        weights = {
            group_name: len(self._alive_workers_locked(group_name)) for group_name in group_names
        }
        total_weight = sum(weights.values())
        for group_name, weight in weights.items():
            self._smooth_weights[group_name] += weight
        selected = max(
            group_names,
            key=lambda group_name: (
                self._smooth_weights[group_name],
                -list(self._groups).index(group_name),
            ),
        )
        self._smooth_weights[selected] -= total_weight
        return selected

    def _ordered_workers_locked(
        self,
        group_name: str,
    ) -> list[_WorkerState]:
        workers = self._alive_workers_locked(group_name)
        cursor = self._group_cursors[group_name]
        count = self._groups[group_name].process_count
        return sorted(
            workers,
            key=lambda worker: (
                worker.outstanding,
                (worker.index - cursor) % count,
            ),
        )

    def _result_loop(self) -> None:
        while not self._result_stop.is_set():
            try:
                item = self._result_queue.get(timeout=0.1)
            except Empty:
                continue

            task_key = item.task_key
            assignment = (item.worker_id, item.generation)

            with self._lock:
                if self._assignments.get(task_key) != assignment:
                    log.debug(
                        "worker_result_ignored task_key=%s reason=stale_assignment",
                        task_key,
                    )
                    continue
                self._assignments.pop(task_key, None)
                future = self._futures.pop(task_key, None)
                worker = self._workers.get(assignment[0])
                if worker is not None and worker.outstanding > 0:
                    worker.outstanding -= 1

            if future is None or future.done():
                continue
            try:
                if item.ok:
                    future.set_result(item.payload)
                    log.debug(
                        "worker_result_completed task_key=%s worker_id=%s generation=%s",
                        task_key,
                        assignment[0],
                        assignment[1],
                    )
                else:
                    log.warning(
                        "worker_result_failed task_key=%s worker_id=%s generation=%s error=%s",
                        task_key,
                        assignment[0],
                        assignment[1],
                        item.error_message or "Unknown worker error",
                    )
                    future.set_exception(
                        PlatformError(
                            _load_error_code(item.error_code),
                            item.error_message or "Unknown worker error",
                        )
                    )
            except Exception as exc:
                log.error(
                    "worker_result_processing_failed task_key=%s error=%s",
                    task_key,
                    exc,
                    exc_info=True,
                )
                if not future.done():
                    future.set_exception(exc)

    def _supervisor_loop(self) -> None:
        log.info("pool_supervisor_started")
        while not self._supervisor_stop.wait(0.05):
            now = time.monotonic()
            for worker in list(self._workers.values()):
                if self._closing:
                    log.info("pool_supervisor_stopped")
                    return
                if worker.alive and not worker.process.is_alive():
                    self._handle_worker_exit(worker, now)
                elif (
                    not worker.alive
                    and worker.next_restart_at is not None
                    and now >= worker.next_restart_at
                ):
                    self._restart_worker(worker)

    def _handle_worker_exit(
        self,
        worker: _WorkerState,
        now: float,
    ) -> None:
        with self._lock:
            if not worker.alive:
                return
            worker.alive = False
            self._smooth_weights[worker.group.name] = 0
            worker.process.join(timeout=0)
            failed = self._take_worker_futures_locked(worker)

        log.warning(
            "worker_exited worker_id=%s generation=%s pending_failed=%s mode=%s",
            worker.worker_id,
            worker.generation,
            len(failed),
            worker.group.safety.mode,
        )
        self._fail_futures(
            failed,
            PlatformError(
                ErrorCode.WORKER_UNAVAILABLE,
                f"Worker '{worker.worker_id}' exited unexpectedly",
            ),
        )

        mode = worker.group.safety.mode
        if mode == "shutdown":
            self._shutdown_after_worker_failure(worker)
        elif mode == "restart":
            self._schedule_restart(worker, now)

    def _take_worker_futures_locked(
        self,
        worker: _WorkerState,
    ) -> list[Future[TaskResult]]:
        failed: list[Future[TaskResult]] = []
        assignment = (worker.worker_id, worker.generation)
        for task_key, owner in list(self._assignments.items()):
            if owner != assignment:
                continue
            self._assignments.pop(task_key, None)
            future = self._futures.pop(task_key, None)
            if future is not None:
                failed.append(future)
        worker.outstanding = 0
        return failed

    def _schedule_restart(
        self,
        worker: _WorkerState,
        now: float,
    ) -> None:
        policy = worker.group.safety
        cutoff = now - policy.restart_window_seconds
        while worker.restart_times and worker.restart_times[0] < cutoff:
            worker.restart_times.popleft()
        if len(worker.restart_times) >= policy.restart_limit:
            worker.next_restart_at = None
            log.warning(
                "worker_degraded worker_id=%s generation=%s reason=restart_limit",
                worker.worker_id,
                worker.generation,
            )
            return
        worker.next_restart_at = now + policy.restart_backoff_seconds
        log.warning(
            "worker_restart_scheduled worker_id=%s generation=%s delay=%s",
            worker.worker_id,
            worker.generation,
            policy.restart_backoff_seconds,
        )

    def _restart_worker(self, worker: _WorkerState) -> None:
        worker.next_restart_at = None
        worker.restart_times.append(time.monotonic())
        log.info(
            "worker_restart_started worker_id=%s previous_generation=%s",
            worker.worker_id,
            worker.generation,
        )
        try:
            self._launch_worker(worker)
            deadline = time.monotonic() + self._startup_timeout_seconds
            self._await_worker_ready(
                worker,
                deadline,
                stop_event=self._supervisor_stop,
            )
            with self._lock:
                self._smooth_weights[worker.group.name] = 0
            log.info(
                "worker_restart_completed worker_id=%s generation=%s",
                worker.worker_id,
                worker.generation,
            )
        except Exception as exc:
            log.error(
                "worker_restart_failed worker_id=%s generation=%s error=%s",
                worker.worker_id,
                worker.generation,
                exc,
                exc_info=True,
            )
            self._stop_process(worker, terminate_after=0)
            if not self._closing:
                self._schedule_restart(worker, time.monotonic())

    def _shutdown_after_worker_failure(
        self,
        failed_worker: _WorkerState,
    ) -> None:
        log.error(
            "pool_shutdown_after_worker_failure worker_id=%s generation=%s",
            failed_worker.worker_id,
            failed_worker.generation,
        )
        with self._lock:
            self._accepting = False
            self._closing = True
        self._supervisor_stop.set()
        self._result_stop.set()
        for worker in self._workers.values():
            if worker is failed_worker:
                continue
            self._stop_process(worker, terminate_after=0)
        self._fail_all_pending(
            PlatformError(
                ErrorCode.WORKER_UNAVAILABLE,
                f"Pool shut down after worker '{failed_worker.worker_id}' failed",
            )
        )
        self._log_manager.stop()

    def _fail_all_pending(self, exc: Exception) -> None:
        with self._lock:
            futures = list(self._futures.values())
            self._futures.clear()
            self._assignments.clear()
            for worker in self._workers.values():
                worker.outstanding = 0
        self._fail_futures(futures, exc)

    @staticmethod
    def _fail_futures(
        futures: list[Future[TaskResult]],
        exc: Exception,
    ) -> None:
        for future in futures:
            if not future.done():
                future.set_exception(exc)


__all__ = ["ProcessPoolStrategy"]
