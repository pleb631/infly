from __future__ import annotations

import copy as _copy
import threading
import time
import uuid
from concurrent.futures import Future

from infly.core.contracts import (
    TaskQueryResponse,
    TaskRecord,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from infly.core.errors import ErrorCode, PlatformError
from infly.core.ports import ExecutionStrategy, TaskBackend
from infly.runtime.config import SchedulerConfig
from infly.runtime.log import get_logger
from infly.runtime.observability import (
    HealthStatus,
    RuntimeInstrumentation,
    SchedulerHealthSnapshot,
    StrategyHealthSnapshot,
)
from infly.runtime.task_backend import InMemoryTaskBackend

log = get_logger()

_TERMINAL_READ_STATUSES = {TaskStatus.COMPLETED, TaskStatus.FAILED}
_WAIT_POLL_INTERVAL_SECONDS = 0.1


class TaskScheduler:
    def __init__(
        self,
        strategy: ExecutionStrategy,
        *,
        backend: TaskBackend | None = None,
        scheduler_config: SchedulerConfig | None = None,
        instrumentation: RuntimeInstrumentation | None = None,
    ) -> None:
        self._strategy = strategy
        self._scheduler_config = scheduler_config or SchedulerConfig()
        self._instrumentation = instrumentation or RuntimeInstrumentation()
        self._backend = (
            backend
            if backend is not None
            else InMemoryTaskBackend(
                max_retained_terminal_tasks=self._scheduler_config.max_retained_terminal_tasks
            )
        )
        self._outstanding_slots = threading.BoundedSemaphore(
            self._scheduler_config.max_outstanding_tasks
        )
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lifecycle_lock = threading.Lock()
        self._threads_lock = threading.Lock()
        self._outstanding_task_ids: set[str] = set()
        self._outstanding_lock = threading.Lock()
        self._accepting = True
        self._strategy_closed = False
        self._closed = False

    @property
    def backend(self) -> TaskBackend:
        return self._backend

    @property
    def instrumentation(self) -> RuntimeInstrumentation:
        return self._instrumentation

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                log.debug("scheduler_start_rejected reason=closed")
                raise PlatformError(
                    ErrorCode.INVALID_STATE,
                    "Scheduler is closed and cannot be started again.",
                )
            with self._threads_lock:
                if self._threads:
                    log.debug("scheduler_start_skipped reason=already_started")
                    return
            self._stop.clear()
            for index in range(self._scheduler_config.num_threads):
                name = f"scheduler-{index}"
                thread = threading.Thread(
                    target=self._worker_loop,
                    name=name,
                    daemon=True,
                    args=(name,),
                )
                with self._threads_lock:
                    self._threads.append(thread)
                thread.start()
            with self._threads_lock:
                worker_count = len(self._threads)
            log.info(
                "scheduler_started workers=%s max_outstanding=%s",
                worker_count,
                self._scheduler_config.max_outstanding_tasks,
            )

    def stop(self, timeout: float = 2.0) -> None:
        if timeout < 0:
            raise PlatformError(
                ErrorCode.INVALID_ARGUMENT,
                "timeout must be greater than or equal to zero.",
            )
        with self._lifecycle_lock:
            self._stop.set()
            self._accepting = False
            self._closed = True
            with self._threads_lock:
                threads = list(self._threads)
            log.info("scheduler_stopping workers=%s", len(threads))
            if not self._strategy_closed:
                self._strategy.close()
                self._strategy_closed = True
            self._notify_waiters()
            deadline = time.monotonic() + timeout
            for thread in threads:
                remaining = max(0.0, deadline - time.monotonic())
                thread.join(remaining)
            with self._threads_lock:
                self._threads = [
                    thread for thread in self._threads if thread.is_alive()
                ]
                has_threads = bool(self._threads)
            if not has_threads:
                self._fail_pending_tasks()
            log.info("scheduler_stopped")

    def _fail_pending_tasks(self) -> None:
        with self._outstanding_lock:
            outstanding_task_ids = set(self._outstanding_task_ids)
        pending_records = [
            record
            for record in self._backend.list_all()
            if record.task_id in outstanding_task_ids
            and record.status == TaskStatus.PENDING
        ]
        for record in pending_records:
            self._backend.update_status(
                record.task_id,
                TaskStatus.FAILED,
                error_code=ErrorCode.WORKER_UNAVAILABLE,
                error_message="Scheduler stopped before task execution started.",
            )
            self._release_outstanding_slot(record.task_id)
        if pending_records:
            self._notify_waiters()

    def submit(self, request: TaskRequest, *, priority: int = 0) -> str:
        with self._lifecycle_lock:
            if not self._accepting:
                log.warning(
                    "scheduler_submission_rejected task_key=%s handler=%s reason=closed",
                    request.task_key,
                    request.handler_name,
                )
                raise PlatformError(
                    ErrorCode.INVALID_STATE,
                    "Scheduler is closed and no longer accepts submissions.",
                )

            if not self._outstanding_slots.acquire(blocking=False):
                log.warning(
                    "scheduler_overloaded task_key=%s handler=%s limit=%s",
                    request.task_key,
                    request.handler_name,
                    self._scheduler_config.max_outstanding_tasks,
                )
                raise PlatformError(
                    ErrorCode.OVERLOADED,
                    "Scheduler has reached its outstanding task limit "
                    f"({self._scheduler_config.max_outstanding_tasks}).",
                )

            task_id = str(uuid.uuid4())
            with self._outstanding_lock:
                self._outstanding_task_ids.add(task_id)
            try:
                self._backend.submit(
                    TaskRecord(task_id=task_id, request=request),
                    priority=priority,
                )
            except Exception as exc:
                self._release_outstanding_slot(task_id)
                log.error(
                    "task_submission_failed task_key=%s handler=%s error=%s",
                    request.task_key,
                    request.handler_name,
                    exc,
                    exc_info=True,
                )
                raise

            log.debug(
                "scheduler_task_accepted task_id=%s task_key=%s handler=%s priority=%s",
                task_id,
                request.task_key,
                request.handler_name,
                priority,
            )
            self._instrumentation.record_submitted(task_id, request)
            self._notify_waiters()
            return task_id

    def submit_and_wait(
        self,
        request: TaskRequest,
        *,
        priority: int = 0,
        timeout_seconds: float | None = None,
        consume: bool = False,
    ) -> TaskResult:
        self.start()
        task_id = self.submit(request, priority=priority)
        try:
            terminal = self.query(
                task_id,
                wait=True,
                timeout_seconds=timeout_seconds,
                consume=consume,
            )
        except PlatformError as exc:
            if exc.code == ErrorCode.TIMEOUT:
                raise self._timeout_error(
                    task_id,
                    "complete",
                    api_name="submit_and_wait",
                ) from exc
            raise

        if terminal.status == TaskStatus.COMPLETED:
            if terminal.result is None:
                raise PlatformError(
                    ErrorCode.INTERNAL_ERROR,
                    "Task completed without a result.",
                )
            return terminal.result
        raise PlatformError(
            terminal.error_code or ErrorCode.INTERNAL_ERROR,
            terminal.error_message or "Task execution failed.",
        )

    def query(
        self,
        task_id: str,
        *,
        wait: bool = False,
        timeout_seconds: float | None = None,
        consume: bool = False,
    ) -> TaskQueryResponse:
        record = self._backend.get(task_id, copy=False)
        if record is None:
            log.warning("task_query_failed task_id=%s reason=not_found", task_id)
            raise PlatformError(ErrorCode.NOT_FOUND, f"Task '{task_id}' not found.")
        if record.status in _TERMINAL_READ_STATUSES:
            return self._read_terminal_response(task_id, consume=consume)
        if not wait:
            return TaskQueryResponse.from_record(_copy.deepcopy(record))

        return self._wait_for_terminal_response(
            task_id,
            timeout_seconds=timeout_seconds,
            consume=consume,
        )

    def _wait_for_terminal_response(
        self,
        task_id: str,
        *,
        timeout_seconds: float | None,
        consume: bool,
    ) -> TaskQueryResponse:
        deadline = None
        if timeout_seconds is not None:
            if timeout_seconds < 0:
                raise PlatformError(
                    ErrorCode.INVALID_ARGUMENT,
                    "timeout_seconds must be greater than or equal to zero.",
                )
            deadline = time.monotonic() + timeout_seconds

        with self._condition:
            while True:
                record = self._backend.get(task_id, copy=False)
                if record is None:
                    raise PlatformError(
                        ErrorCode.NOT_FOUND, f"Task '{task_id}' not found."
                    )
                if record.status in _TERMINAL_READ_STATUSES:
                    return self._read_terminal_response(
                        task_id,
                        consume=consume,
                    )
                if deadline is not None:
                    remaining = self._remaining_time(deadline)
                    if remaining <= 0:
                        phase = (
                            "start"
                            if record.status == TaskStatus.PENDING
                            else "complete"
                        )
                        raise self._timeout_error(task_id, phase)
                    self._condition.wait(
                        timeout=min(remaining, _WAIT_POLL_INTERVAL_SECONDS)
                    )
                else:
                    self._condition.wait()

    def _read_terminal_response(
        self,
        task_id: str,
        *,
        consume: bool,
    ) -> TaskQueryResponse:
        record = self._backend.read(task_id, consume=consume)
        if record is None:
            log.warning(
                "task_query_failed task_id=%s reason=consumed_or_evicted",
                task_id,
            )
            raise PlatformError(ErrorCode.NOT_FOUND, f"Task '{task_id}' not found.")
        return TaskQueryResponse.from_record(record)

    @staticmethod
    def _remaining_time(deadline: float) -> float:
        return deadline - time.monotonic()

    @staticmethod
    def _timeout_error(
        task_id: str,
        phase: str,
        *,
        api_name: str | None = None,
    ) -> PlatformError:
        if api_name is None:
            message = f"Timed out waiting for task '{task_id}' to {phase}."
        else:
            message = (
                f"{api_name} timed out for task '{task_id}' while waiting for task "
                f"to {phase}."
            )
        return PlatformError(ErrorCode.TIMEOUT, message)

    def _pull_task_or_none(self) -> str | None:
        with self._condition:
            while True:
                if self._stop.is_set():
                    return None
                task_id = self._backend.pull()
                if task_id is not None:
                    return task_id
                self._condition.wait()

    def _worker_loop(self, thread_name: str) -> None:
        log.info(f"{thread_name} started")
        try:
            while True:
                try:
                    task_id = self._pull_task_or_none()
                except Exception as exc:
                    log.error(
                        "worker_loop_pull_failed thread=%s error=%s",
                        thread_name,
                        exc,
                        exc_info=(type(exc), exc, exc.__traceback__),
                    )
                    if self._stop.is_set():
                        return
                    time.sleep(_WAIT_POLL_INTERVAL_SECONDS)
                    continue
                if task_id is None:
                    return
                try:
                    self._run_task(task_id)
                except Exception as exc:
                    self._handle_worker_loop_task_failure(task_id, exc)
        finally:
            self._worker_exited(threading.current_thread())
            log.info(f"{thread_name} stopped")

    def _worker_exited(self, thread: threading.Thread) -> None:
        with self._threads_lock:
            is_last_worker = all(
                other is thread or not other.is_alive() for other in self._threads
            )
            if self._stop.is_set() and is_last_worker:
                self._fail_pending_tasks()
            if thread in self._threads:
                self._threads.remove(thread)

    def _run_task(self, task_id: str) -> None:
        record = self._backend.get(task_id, copy=False)
        if record is None:
            log.warning("task_execution_skipped task_id=%s reason=not_found", task_id)
            return
        if record.status != TaskStatus.PENDING:
            log.debug(
                "task_execution_skipped task_id=%s reason=status status=%s",
                task_id,
                record.status,
            )
            return

        self._backend.update_status(task_id, TaskStatus.RUNNING)
        self._instrumentation.record_started(task_id, record.request)
        log.debug(
            "task_execution_started task_id=%s task_key=%s handler=%s",
            task_id,
            record.request.task_key,
            record.request.handler_name,
        )
        try:
            execution_future = self._strategy.execute(record.request)
        except Exception as exc:
            self._fail_task(task_id, exc)
            return

        execution_future.add_done_callback(
            lambda completed: self._complete_task(task_id, completed)
        )

    def _handle_worker_loop_task_failure(
        self,
        task_id: str,
        exc: Exception,
    ) -> None:
        log.error(
            "worker_loop_task_failed task_id=%s error=%s",
            task_id,
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        try:
            record = self._backend.get(task_id, copy=False)
        except Exception as lookup_exc:
            log.error(
                "worker_loop_task_lookup_failed task_id=%s error=%s",
                task_id,
                lookup_exc,
                exc_info=(type(lookup_exc), lookup_exc, lookup_exc.__traceback__),
            )
            record = None
        try:
            if record is not None and record.status not in _TERMINAL_READ_STATUSES:
                self._backend.update_status(
                    task_id,
                    TaskStatus.FAILED,
                    error_code=ErrorCode.INTERNAL_ERROR,
                    error_message=f"Worker loop failed: {exc}",
                )
                self._instrumentation.record_failed(
                    task_id,
                    record.request,
                    error_code=ErrorCode.INTERNAL_ERROR,
                    error_message=f"Worker loop failed: {exc}",
                )
        except Exception as update_exc:
            log.error(
                "worker_loop_task_update_failed task_id=%s error=%s",
                task_id,
                update_exc,
                exc_info=(type(update_exc), update_exc, update_exc.__traceback__),
            )
        finally:
            self._finish_execution(task_id)

    @staticmethod
    def _execution_error_code(exc: Exception) -> ErrorCode:
        if isinstance(exc, PlatformError) and exc.code == ErrorCode.WORKER_UNAVAILABLE:
            return ErrorCode.WORKER_UNAVAILABLE
        return ErrorCode.INTERNAL_ERROR

    def _fail_task(self, task_id: str, exc: Exception) -> None:
        record = self._backend.get(task_id, copy=False)
        log.error(
            "task_execution_failed task_id=%s error=%s",
            task_id,
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        error_code = self._execution_error_code(exc)
        error_message = f"Execution failed: {exc}"
        self._backend.update_status(
            task_id,
            TaskStatus.FAILED,
            error_code=error_code,
            error_message=error_message,
        )
        if record is not None:
            self._instrumentation.record_failed(
                task_id,
                record.request,
                error_code=error_code,
                error_message=error_message,
            )
        self._finish_execution(task_id)

    def _finish_execution(self, task_id: str) -> None:
        self._release_outstanding_slot(task_id)
        self._notify_waiters()

    def _release_outstanding_slot(self, task_id: str) -> None:
        with self._outstanding_lock:
            if task_id not in self._outstanding_task_ids:
                return
            self._outstanding_task_ids.remove(task_id)
        self._outstanding_slots.release()

    def _complete_task(
        self,
        task_id: str,
        execution_future: Future[TaskResult],
    ) -> None:
        record = self._backend.get(task_id, copy=False)
        try:
            result = execution_future.result()
        except Exception as exc:
            log.error(
                "task_execution_failed task_id=%s error=%s",
                task_id,
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            self._backend.update_status(
                task_id,
                TaskStatus.FAILED,
                error_code=self._execution_error_code(exc),
                error_message=f"Execution failed: {exc}",
            )
            if record is not None:
                self._instrumentation.record_failed(
                    task_id,
                    record.request,
                    error_code=self._execution_error_code(exc),
                    error_message=f"Execution failed: {exc}",
                )
        else:
            self._backend.update_status(
                task_id,
                TaskStatus.COMPLETED,
                result=result,
            )
            if record is not None:
                self._instrumentation.record_completed(task_id, record.request)
            log.debug("task_execution_completed task_id=%s", task_id)
        finally:
            self._finish_execution(task_id)

    def _notify_waiters(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def health_snapshot(self) -> SchedulerHealthSnapshot:
        with self._threads_lock:
            worker_threads = len(self._threads)

        status_counts = {
            task_status.value: 0
            for task_status in TaskStatus
        }
        for record in self._backend.list_all():
            status_counts[record.status.value] = (
                status_counts.get(record.status.value, 0) + 1
            )

        strategy_snapshot = self._strategy_health_snapshot()
        status = HealthStatus.OK
        if self._closed or (worker_threads == 0 and not self._accepting):
            status = HealthStatus.DOWN
        elif strategy_snapshot is not None and strategy_snapshot.status != HealthStatus.OK:
            status = strategy_snapshot.status

        with self._outstanding_lock:
            outstanding_tasks = len(self._outstanding_task_ids)

        return SchedulerHealthSnapshot(
            status=status,
            accepting=self._accepting,
            started=worker_threads > 0,
            closed=self._closed,
            worker_threads=worker_threads,
            outstanding_tasks=outstanding_tasks,
            max_outstanding_tasks=self._scheduler_config.max_outstanding_tasks,
            backend_status_counts=status_counts,
            strategy=strategy_snapshot,
        )

    def _strategy_health_snapshot(self) -> StrategyHealthSnapshot | None:
        snapshot_fn = getattr(self._strategy, "health_snapshot", None)
        if snapshot_fn is None:
            return None
        snapshot = snapshot_fn()
        if isinstance(snapshot, StrategyHealthSnapshot):
            return snapshot
        return None


__all__ = ["TaskScheduler"]
