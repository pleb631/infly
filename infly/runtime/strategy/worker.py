"""Worker-process protocol and execution loop."""

from __future__ import annotations

import os
import sys
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from multiprocessing import Queue
from threading import RLock
from typing import Any

from infly.core.contracts import TaskResult
from infly.core.errors import ErrorCode
from infly.runtime.config import WorkerGroup
from infly.runtime.log import LoggingSettings, get_logger, log_context


def dump_error_code(code: object) -> object:
    return code.value if isinstance(code, ErrorCode) else code


def load_error_code(code: object) -> ErrorCode:
    if isinstance(code, ErrorCode):
        return code
    try:
        return ErrorCode(code)
    except Exception:
        return ErrorCode.INTERNAL_ERROR


def restore_parent_import_path(parent_sys_path: list[str], parent_cwd: str) -> None:
    """Restore the parent's import roots in a spawned worker."""
    missing = [
        parent_cwd if entry == "" else entry
        for entry in parent_sys_path
        if (parent_cwd if entry == "" else entry) not in sys.path
    ]
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
    task_id: str
    worker_id: str
    generation: int
    payload: TaskResult | None = None
    error_code: ErrorCode | object | None = None
    error_message: str | None = None


@dataclass(slots=True)
class WorkerState:
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
    retiring: bool = False
    restart_times: deque[float] = field(default_factory=deque)
    next_restart_at: float | None = None
    # Process and queue handles are shared by the supervisor and callers of
    # ``ProcessPoolStrategy.close``.  Keep their lifecycle separate from the
    # pool-wide scheduling lock so a slow process operation cannot block
    # request dispatch for every worker.
    lifecycle_lock: Any = field(default_factory=RLock, repr=False)

    def is_running(self) -> bool:
        """Return whether the current process is alive without racing teardown."""
        with self.lifecycle_lock:
            process = self.process
            if not self.alive or process is None:
                return False
            try:
                return bool(process.is_alive())
            except Exception:
                # ``multiprocessing.Process.close`` makes later attribute access
                # raise ValueError. A concurrent shutdown therefore means the
                # worker is no longer eligible for routing.
                return False

    def is_routable(self) -> bool:
        """Return whether dispatch can use this worker without waiting on lifecycle work.

        Request dispatch runs while holding the pool-state lock.  A process
        launch or teardown holds ``lifecycle_lock`` and may take appreciable
        time, so waiting for that lock here would pause submission to every
        worker in the pool.  Treat a contended worker as temporarily
        unavailable; the router can select another worker or return the usual
        unavailable error while its lifecycle operation completes.
        """
        if not self.alive or self.retiring:
            return False
        if not self.lifecycle_lock.acquire(blocking=False):
            return False
        try:
            process = self.process
            if process is None:
                return False
            try:
                return bool(process.is_alive())
            except Exception:
                return False
        finally:
            self.lifecycle_lock.release()


def run_worker_loop(
    *,
    worker_id: str,
    generation: int,
    task_queue: Queue,
    result_queue: Queue,
    lifecycle_queue: Queue,
    registry: Any,
    environment: Mapping[str, str],
    parent_sys_path: list[str],
    parent_cwd: str,
    log_queue: Queue,
    log_settings: LoggingSettings,
    setup_logging: Callable[..., None],
    set_process_title: Callable[[str], Any],
    restore_import_path: Callable[[list[str], str], None],
    executor_type: Callable[[Any], Any],
) -> None:
    """Preload handlers and execute requests in one child process.

    Runtime dependencies are supplied by the strategy's small compatibility
    adapter.  This keeps the child-process boundary independently testable.
    """
    setup_logging(log_queue, settings=log_settings)
    set_process_title(f"INFLY::{worker_id}")
    log = get_logger(name=worker_id, category="worker")
    log.info("worker_started worker_id=%s generation=%s", worker_id, generation)
    try:
        restore_import_path(parent_sys_path, parent_cwd)
        os.environ.update(environment)
        executor = executor_type(registry)
        with log_context(name=worker_id, category="worker"):
            executor.preload()
        lifecycle_queue.put(WorkerLifecycleMessage("READY", worker_id, generation))
        log.info("worker_ready worker_id=%s generation=%s", worker_id, generation)
    except Exception as exc:
        log.error(
            "worker_startup_failed worker_id=%s generation=%s error=%s",
            worker_id,
            generation,
            exc,
            exc_info=True,
        )
        lifecycle_queue.put(WorkerLifecycleMessage("STARTUP_FAILED", worker_id, generation, str(exc)))
        return

    while True:
        request = task_queue.get()
        if request is None:
            log.info("worker_stopped worker_id=%s generation=%s", worker_id, generation)
            return
        try:
            log.debug(
                "worker_request_started worker_id=%s generation=%s task_id=%s handler=%s",
                worker_id,
                generation,
                request.task_id,
                request.handler_name,
            )
            with log_context(name=worker_id, category="worker"):
                result = executor.execute(request)
            result_queue.put(WorkerResultMessage(True, request.task_id, worker_id, generation, payload=result))
            log.debug(
                "worker_request_completed worker_id=%s generation=%s task_id=%s", worker_id, generation, request.task_id
            )
        except Exception as exc:
            log.error(
                "worker_request_failed worker_id=%s generation=%s task_id=%s error=%s",
                worker_id,
                generation,
                request.task_id,
                exc,
                exc_info=True,
            )
            result_queue.put(
                WorkerResultMessage(
                    False,
                    request.task_id,
                    worker_id,
                    generation,
                    error_code=dump_error_code(getattr(exc, "code", ErrorCode.INTERNAL_ERROR)),
                    error_message=str(exc),
                )
            )
