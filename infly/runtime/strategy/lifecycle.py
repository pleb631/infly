"""Child-process lifecycle management for the process-pool strategy."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Iterable
from contextlib import suppress
from queue import Empty
from typing import Any

from infly.core.errors import ErrorCode, PlatformError
from infly.core.handlers import HandlerDefinition
from infly.runtime.log import LoggingSettings, MainLogManager, get_logger
from infly.runtime.registry import HandlerRegistry
from infly.runtime.strategy.worker import WorkerState

log = get_logger("infly")


class WorkerProcessController:
    """Start, observe, and clean up worker processes.

    The pool owns policy decisions; this controller owns the mechanics of a
    single worker process and its queues.
    """

    def __init__(
        self,
        *,
        registry: HandlerRegistry,
        mp_context: Any,
        queue_factory: Callable[..., Any],
        result_queue: Any,
        log_manager: MainLogManager,
        log_settings: LoggingSettings,
        worker_target: Callable[..., None],
        startup_timeout_seconds: float,
    ) -> None:
        self._registry = registry
        self._mp_context = mp_context
        self._queue_factory = queue_factory
        self._result_queue = result_queue
        self._log_manager = log_manager
        self._log_settings = log_settings
        self._worker_target = worker_target
        self._startup_timeout_seconds = startup_timeout_seconds

    def launch(self, worker: WorkerState) -> None:
        with worker.lifecycle_lock:
            self._launch_locked(worker)

    def _launch_locked(self, worker: WorkerState) -> None:
        worker.generation += 1
        worker.alive = False
        worker.outstanding = 0
        worker.next_restart_at = None
        worker.task_queue = self._queue_factory()
        worker.lifecycle_queue = self._queue_factory(1)
        child_registry = self._build_registry(worker)
        log.info("worker_launching worker_id=%s generation=%s", worker.worker_id, worker.generation)
        worker.process = self._mp_context.Process(
            target=self._worker_target,
            args=(
                worker.worker_id,
                worker.generation,
                worker.task_queue,
                self._result_queue,
                worker.lifecycle_queue,
                child_registry,
                worker.group.environment,
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

    def cleanup_failed_start(self, worker: WorkerState) -> None:
        """Release resources from a failed launch without retiring the worker.

        A restart failure remains eligible for the manager's backoff policy, so
        it must not use ``dispose`` (which marks a worker permanently retired).
        """
        with worker.lifecycle_lock:
            self._stop_locked(worker, terminate_after=0)
            self._close_worker_queues_locked(worker)

    def await_initial_startup(self, workers: Iterable[WorkerState]) -> None:
        deadline = time.monotonic() + self._startup_timeout_seconds
        for worker in workers:
            self.await_ready(worker, deadline)

    def await_ready(
        self,
        worker: WorkerState,
        deadline: float,
        stop_event: Any | None = None,
    ) -> None:
        while True:
            if stop_event is not None and stop_event.is_set():
                raise PlatformError(ErrorCode.INTERNAL_ERROR, f"Worker '{worker.worker_id}' startup was interrupted")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PlatformError(
                    ErrorCode.INTERNAL_ERROR,
                    f"Worker pool startup timed out waiting for '{worker.worker_id}'",
                )
            try:
                with worker.lifecycle_lock:
                    lifecycle_queue = worker.lifecycle_queue
                if lifecycle_queue is None:
                    raise PlatformError(
                        ErrorCode.INTERNAL_ERROR,
                        f"Worker '{worker.worker_id}' stopped during startup",
                    )
                message = lifecycle_queue.get(timeout=min(remaining, 0.1))
                break
            except Empty:
                with worker.lifecycle_lock:
                    process_alive = self._is_process_alive(worker.process)
                if not process_alive:
                    raise PlatformError(
                        ErrorCode.INTERNAL_ERROR,
                        f"Worker '{worker.worker_id}' exited during startup",
                    ) from Empty

        if message.kind != "READY" or message.generation != worker.generation:
            error_message = message.error_message or "unknown error"
            log.error(
                "worker_startup_failed worker_id=%s generation=%s error=%s",
                worker.worker_id,
                worker.generation,
                error_message,
            )
            raise PlatformError(
                ErrorCode.INTERNAL_ERROR,
                f"Worker '{worker.worker_id}' startup failed: {error_message}",
            )
        with worker.lifecycle_lock:
            if not self._is_process_alive(worker.process):
                raise PlatformError(
                    ErrorCode.INTERNAL_ERROR,
                    f"Worker '{worker.worker_id}' exited during startup",
                )
            worker.alive = True
        log.info("worker_ready worker_id=%s generation=%s", worker.worker_id, worker.generation)

    def dispose(self, workers: Iterable[WorkerState]) -> None:
        for worker in workers:
            with worker.lifecycle_lock:
                worker.retiring = True
                worker.next_restart_at = None
                if self._is_process_alive(worker.process) and worker.task_queue is not None:
                    with suppress(Exception):
                        worker.task_queue.put(None, timeout=0.2)
                self._stop_locked(worker, terminate_after=1)
                self._close_worker_queues_locked(worker)

    def close_queues(self, workers: Iterable[WorkerState]) -> None:
        for worker in workers:
            self.close_worker_queues(worker)
        self.close_queue(self._result_queue)

    def close_worker_queues(self, worker: WorkerState) -> None:
        with worker.lifecycle_lock:
            self._close_worker_queues_locked(worker)

    def _close_worker_queues_locked(self, worker: WorkerState) -> None:
        self.close_queue(getattr(worker, "task_queue", None))
        self.close_queue(getattr(worker, "lifecycle_queue", None))
        worker.task_queue = None
        worker.lifecycle_queue = None

    def request_graceful_stop(self, worker: WorkerState) -> None:
        """Ask a live worker to exit without racing its process teardown."""
        with worker.lifecycle_lock:
            if not self._is_process_alive(worker.process) or worker.task_queue is None:
                return
            with suppress(Exception):
                worker.task_queue.put(None, timeout=0.2)

    def stop(self, worker: WorkerState, *, terminate_after: float) -> None:
        with worker.lifecycle_lock:
            self._stop_locked(worker, terminate_after=terminate_after)

    def _stop_locked(self, worker: WorkerState, *, terminate_after: float) -> None:
        process = worker.process
        if process is None:
            return
        with suppress(Exception):
            process.join(timeout=terminate_after)
        is_alive = self._is_process_alive(process)
        if is_alive:
            log.warning(
                "worker_terminating worker_id=%s generation=%s pid=%s",
                worker.worker_id,
                worker.generation,
                getattr(process, "pid", None),
            )
            with suppress(Exception):
                process.terminate()
            with suppress(Exception):
                process.join(timeout=1)
        close = getattr(process, "close", None)
        if close is not None:
            with suppress(Exception):
                close()
        worker.process = None
        worker.alive = False
        log.info("worker_stopped worker_id=%s generation=%s", worker.worker_id, worker.generation)

    @staticmethod
    def _is_process_alive(process: Any) -> bool:
        if process is None:
            return False
        try:
            return bool(process.is_alive())
        except Exception:
            return False

    @staticmethod
    def close_queue(queue: object | None) -> None:
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

    def _build_registry(self, worker: WorkerState) -> HandlerRegistry:
        child_registry = HandlerRegistry()
        context = {"group_name": worker.group.name, "worker_id": worker.worker_id}
        for handler_name in worker.handler_names:
            definition = self._registry.get(handler_name)
            child_registry.add(HandlerDefinition.with_runtime_context(definition, runtime_context=context))
        return child_registry
