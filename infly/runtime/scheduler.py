from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import Future

from infly.core.contracts import (
    InferenceRequest,
    InferenceResult,
    TaskQueryResponse,
    TaskRecord,
    TaskStatus,
)
from infly.core.errors import ErrorCode, PlatformError
from infly.core.ports import ExecutionStrategy, TaskBackend
from infly.runtime.config import SchedulerConfig
from infly.runtime.task_backend import InMemoryTaskBackend
from infly.runtime.log import get_logger

log = get_logger()

class TaskScheduler:
    def __init__(
        self,
        strategy: ExecutionStrategy,
        *,
        backend: TaskBackend | None = None,
        scheduler_config: SchedulerConfig | None = None,
    ) -> None:
        self._strategy = strategy
        self._scheduler_config = scheduler_config or SchedulerConfig()
        self._backend = (
            backend
            if backend is not None
            else InMemoryTaskBackend(
                max_retained_terminal_tasks=(
                    self._scheduler_config.max_retained_terminal_tasks
                )
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
        self._execution_futures: dict[str, Future[InferenceResult]] = {}
        self._outstanding_task_ids: set[str] = set()
        self._outstanding_lock = threading.Lock()
        self._accepting = True
        self._strategy_closed = False
        self._closed = False

    @property
    def backend(self) -> TaskBackend:
        return self._backend

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                log.debug("scheduler_start_rejected reason=closed")
                raise PlatformError(
                    ErrorCode.INVALID_INPUT,
                    "Scheduler is closed and cannot be started again.",
                )
            with self._threads_lock:
                if self._threads:
                    log.debug("scheduler_start_skipped reason=already_started")
                    return
            self._stop.clear()
            for index in range(self._scheduler_config.num_workers):
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
                ErrorCode.INVALID_INPUT,
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
            with self._condition:
                self._condition.notify_all()
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

    def submit(self, request: InferenceRequest, *, priority: int = 0) -> str:
        with self._lifecycle_lock:
            if not self._accepting:
                log.warning(
                    "scheduler_submission_rejected request_id=%s model=%s reason=closed",
                    request.request_id,
                    request.model_name,
                )
                raise PlatformError(
                    ErrorCode.INVALID_INPUT,
                    "Scheduler is closed and no longer accepts submissions.",
                )

            if not self._outstanding_slots.acquire(blocking=False):
                log.warning(
                    "scheduler_overloaded request_id=%s model=%s limit=%s",
                    request.request_id,
                    request.model_name,
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
                    "task_submission_failed request_id=%s model=%s error=%s",
                    request.request_id,
                    request.model_name,
                    exc,
                    exc_info=True,
                )
                raise

            log.debug(
                "scheduler_task_accepted task_id=%s request_id=%s model=%s priority=%s",
                task_id,
                request.request_id,
                request.model_name,
                priority,
            )
            with self._condition:
                self._condition.notify_all()
            return task_id

    def submit_and_wait(
        self,
        request: InferenceRequest,
        *,
        priority: int = 0,
        timeout_seconds: float | None = None,
        consume: bool = False,
    ) -> InferenceResult:
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
            return InferenceResult.model_validate(terminal.result)
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
        record = self._backend.get(task_id)
        if record is None:
            log.warning("task_query_failed task_id=%s reason=not_found", task_id)
            raise PlatformError(ErrorCode.NOT_FOUND, f"Task '{task_id}' not found.")
        if record.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
            return self._read_terminal_response(task_id, consume=consume)
        if not wait:
            return TaskQueryResponse.from_record(record)

        deadline = None
        if timeout_seconds is not None:
            if timeout_seconds < 0:
                raise PlatformError(
                    ErrorCode.INVALID_INPUT,
                    "timeout_seconds must be greater than or equal to zero.",
                )
            deadline = time.monotonic() + timeout_seconds

        with self._condition:
            while True:
                record = self._backend.get(task_id)
                if record is None:
                    raise PlatformError(
                        ErrorCode.NOT_FOUND, f"Task '{task_id}' not found."
                    )
                if record.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
                    return self._read_terminal_response(
                        task_id,
                        consume=consume,
                    )
                execution_future = self._execution_futures.get(task_id)
                if execution_future is not None:
                    break
                if deadline is not None:
                    remaining = self._remaining_time(deadline)
                    if remaining <= 0:
                        raise self._timeout_error(task_id, "start")
                    self._condition.wait(timeout=min(remaining, 0.1))
                else:
                    self._condition.wait()

        try:
            if deadline is None:
                execution_future.result()
            else:
                remaining = self._remaining_time(deadline)
                if remaining <= 0:
                    raise self._timeout_error(task_id, "complete")
                execution_future.result(timeout=remaining)
        except Exception:
            if deadline is not None and time.monotonic() >= deadline:
                raise self._timeout_error(task_id, "complete")

        with self._condition:
            while task_id in self._execution_futures:
                if deadline is not None:
                    remaining = self._remaining_time(deadline)
                    if remaining <= 0:
                        raise self._timeout_error(task_id, "complete")
                    self._condition.wait(timeout=min(remaining, 0.1))
                else:
                    self._condition.wait()

        record = self._backend.get(task_id)
        if record is None:
            raise PlatformError(ErrorCode.NOT_FOUND, f"Task '{task_id}' not found.")
        if record.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
            return self._read_terminal_response(task_id, consume=consume)
        return TaskQueryResponse.from_record(record)

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
                task_id = self._pull_task_or_none()
                if task_id is None:
                    return
                self._run_task(task_id)
        finally:
            self._worker_exited(threading.current_thread())
            log.info(f"{thread_name} stopped")

    def _worker_exited(self, thread: threading.Thread) -> None:
        with self._threads_lock:
            is_last_worker = all(
                other is thread or not other.is_alive()
                for other in self._threads
            )
            if self._stop.is_set() and is_last_worker:
                self._fail_pending_tasks()
            if thread in self._threads:
                self._threads.remove(thread)

    def _run_task(self, task_id: str) -> None:
        record = self._backend.get(task_id)
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
        log.debug(
            "task_execution_started task_id=%s request_id=%s model=%s",
            task_id,
            record.request.request_id,
            record.request.model_name,
        )
        try:
            execution_future = self._strategy.execute(record.request)
        except Exception as exc:
            self._fail_task(task_id, exc)
            return

        with self._condition:
            self._execution_futures[task_id] = execution_future
            execution_future.add_done_callback(
                lambda completed: self._complete_task(task_id, completed)
            )
            self._condition.notify_all()

    @staticmethod
    def _execution_error_code(exc: Exception) -> ErrorCode:
        if isinstance(exc, PlatformError) and exc.code == ErrorCode.WORKER_UNAVAILABLE:
            return ErrorCode.WORKER_UNAVAILABLE
        return ErrorCode.INTERNAL_ERROR

    def _fail_task(self, task_id: str, exc: Exception) -> None:
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
        self._finish_execution(task_id)

    def _finish_execution(self, task_id: str) -> None:
        self._release_outstanding_slot(task_id)
        with self._condition:
            self._execution_futures.pop(task_id, None)
            self._condition.notify_all()

    def _release_outstanding_slot(self, task_id: str) -> None:
        with self._outstanding_lock:
            if task_id not in self._outstanding_task_ids:
                return
            self._outstanding_task_ids.remove(task_id)
        self._outstanding_slots.release()

    def _complete_task(
        self,
        task_id: str,
        execution_future: Future[InferenceResult],
    ) -> None:
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
        else:
            self._backend.update_status(
                task_id,
                TaskStatus.COMPLETED,
                result=result.model_dump(),
            )
            log.debug("task_execution_completed task_id=%s", task_id)
        finally:
            self._finish_execution(task_id)


__all__ = ["TaskScheduler"]
