from __future__ import annotations

import os
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future
from multiprocessing import get_context,Queue
from pydantic import BaseModel, Field
from queue import Empty
from typing import Any
from setproctitle import setproctitle

from infly.core.contracts import InferenceRequest, InferenceResult
from infly.core.errors import ErrorCode, PlatformError
from infly.core.models import ModelDefinition
from infly.runtime.config import WorkerGroup
from infly.runtime.registry import ModelRegistry
from infly.runtime.service import InferenceService
from infly.runtime.log import (
    RoutingQueueListener,
    get_logger,
    setup_worker_logging,
)


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


def _worker_loop(
    worker_id: str,
    generation: int,
    task_queue: Queue,
    result_queue: Queue,
    lifecycle_queue: Queue,
    registry: ModelRegistry,
    environment: dict[str, str],
    device: str,
    parent_sys_path: list[str],
    parent_cwd: str,
    log_queue: Any,
) -> None:
    
    setup_worker_logging(log_queue)
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
        service = InferenceService(registry, log)
        service.preload()
        lifecycle_queue.put(
            {
                "kind": "READY",
                "worker_id": worker_id,
                "generation": generation,
            }
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
            {
                "kind": "STARTUP_FAILED",
                "worker_id": worker_id,
                "generation": generation,
                "error_message": str(exc),
            }
        )
        return

    while True:
        item = task_queue.get()
        if item is None:
            log.info(
                "worker_stopped worker_id=%s generation=%s",
                worker_id,
                generation,
            )
            return
        try:
            request = InferenceRequest.model_validate(item)
            log.debug(
                "worker_request_started worker_id=%s generation=%s request_id=%s model=%s",
                worker_id,
                generation,
                request.request_id,
                request.model_name,
            )
            result = service.predict(request)
            result_queue.put(
                {
                    "ok": True,
                    "payload": result.model_dump(),
                    "request_id": request.request_id,
                    "worker_id": worker_id,
                    "generation": generation,
                }
            )
            log.debug(
                "worker_request_completed worker_id=%s generation=%s request_id=%s",
                worker_id,
                generation,
                request.request_id,
            )
        except Exception as exc:
            request_id = item.get("request_id") if isinstance(item, dict) else None
            log.error(
                "worker_request_failed worker_id=%s generation=%s request_id=%s error=%s",
                worker_id,
                generation,
                request_id,
                exc,
                exc_info=True,
            )
            result_queue.put(
                {
                    "ok": False,
                    "error_code": _dump_error_code(
                        getattr(exc, "code", ErrorCode.INTERNAL_ERROR)
                    ),
                    "error_message": str(exc),
                    "request_id": request_id,
                    "worker_id": worker_id,
                    "generation": generation,
                }
            )


class _WorkerState(BaseModel):
    worker_id: str
    index: int
    group: WorkerGroup
    model_names: tuple[str, ...]
    generation: int = 0
    task_queue: Any = None
    lifecycle_queue: Any = None
    process: Any = None
    alive: bool = False
    outstanding: int = 0
    restart_times: deque[float] = Field(default_factory=deque)
    next_restart_at: float | None = None


log = get_logger("infly")


class EmbeddedProcessPoolStrategy:
    name = "embedded_process_pool"

    def __init__(
        self,
        registry: ModelRegistry,
        worker_groups: list[WorkerGroup],
        *,
        startup_timeout_seconds: float = 300,
    ) -> None:

        log.info(
            "pool_starting groups=%s startup_timeout=%s",
            len(worker_groups),
            startup_timeout_seconds,
        )
        if not worker_groups:
            raise PlatformError(
                ErrorCode.INTERNAL_ERROR,
                "EmbeddedProcessPoolStrategy requires at least one worker group",
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

        self._registry = registry
        self._startup_timeout_seconds = startup_timeout_seconds
        self._groups = {group.name: group for group in worker_groups}
        self._group_models: dict[str, tuple[str, ...]] = {}
        self._model_groups: dict[str, list[str]] = defaultdict(list)
        self._workers: dict[str, _WorkerState] = {}
        self._group_cursors: dict[str, int] = defaultdict(int)
        self._smooth_weights: dict[str, int] = defaultdict(int)
        self._futures: dict[str, Future[InferenceResult]] = {}
        self._assignments: dict[str, tuple[str, int]] = {}
        self._lock = threading.RLock()
        self._mp_context = get_context("spawn")
        self._result_stop = threading.Event()
        self._supervisor_stop = threading.Event()
        self._accepting = True
        self._closing = False
        self._close_complete = False
        self._logging_started = False
        self._logging_stopped = True

        all_model_names = tuple(definition.model_name for definition in registry.list())
        for group in worker_groups:
            model_names = tuple(group.models) if group.models else all_model_names
            for model_name in model_names:
                try:
                    registry.get(model_name)
                except PlatformError as exc:
                    raise PlatformError(
                        ErrorCode.INTERNAL_ERROR,
                        f"WorkerGroup '{group.name}' references missing model "
                        f"'{model_name}'",
                    ) from exc
                self._model_groups[model_name].append(group.name)
            self._group_models[group.name] = model_names
            for index in range(group.process_count):
                worker_id = f"{group.name}_R{index}"
                self._workers[worker_id] = _WorkerState(
                    worker_id=worker_id,
                    index=index,
                    group=group,
                    model_names=model_names,
                )

        self.log_queue: Queue = self._mp_context.Queue()
        self.listener = RoutingQueueListener(self.log_queue)
        try:
            self._result_queue: Queue = self._mp_context.Queue()
            for worker in self._workers.values():
                self._launch_worker(worker)
            self._await_initial_startup()
            self._logging_started = True
            self._logging_stopped = False
            self.listener.start()
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
        request: InferenceRequest,
    ) -> Future[InferenceResult]:
        future: Future[InferenceResult] = Future()
        with self._lock:
            if not self._accepting:
                log.warning(
                    "request_rejected request_id=%s reason=pool_closed",
                    request.request_id,
                )
                future.set_exception(
                    PlatformError(
                        ErrorCode.INTERNAL_ERROR,
                        "EmbeddedProcessPoolStrategy is closed",
                    )
                )
                return future

            if request.request_id in self._futures:
                log.warning(
                    "request_rejected request_id=%s reason=duplicate",
                    request.request_id,
                )
                future.set_exception(
                    PlatformError(
                        ErrorCode.INTERNAL_ERROR,
                        f"Duplicate request_id: {request.request_id}",
                    )
                )
                return future

            group_names = [
                group_name
                for group_name in self._model_groups.get(request.model_name, [])
                if self._alive_workers_locked(group_name)
            ]
            if not group_names:
                log.warning(
                    "request_rejected request_id=%s model=%s reason=no_live_worker",
                    request.request_id,
                    request.model_name,
                )
                future.set_exception(
                    PlatformError(
                        ErrorCode.WORKER_UNAVAILABLE,
                        f"No live worker is deployed for model '{request.model_name}'",
                    )
                )
                return future

            self._futures[request.request_id] = future
            remaining_groups = list(group_names)
            while remaining_groups:
                group_name = self._select_group_locked(remaining_groups)
                for worker in self._ordered_workers_locked(group_name):
                    assignment = (worker.worker_id, worker.generation)
                    self._assignments[request.request_id] = assignment
                    worker.outstanding += 1
                    try:
                        worker.task_queue.put_nowait(request.model_dump())
                    except Exception as exc:
                        worker.outstanding -= 1
                        self._assignments.pop(request.request_id, None)
                        log.warning(
                            "request_assignment_failed request_id=%s worker_id=%s "
                            "generation=%s error=%s",
                            request.request_id,
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
                        "request_assigned request_id=%s model=%s worker_id=%s "
                        "generation=%s",
                        request.request_id,
                        request.model_name,
                        worker.worker_id,
                        worker.generation,
                    )
                    return future
                remaining_groups.remove(group_name)

            self._futures.pop(request.request_id, None)
            log.warning(
                "request_rejected request_id=%s model=%s reason=assignment_failed",
                request.request_id,
                request.model_name,
            )
            future.set_exception(
                PlatformError(
                    ErrorCode.WORKER_UNAVAILABLE,
                    f"Unable to submit request to a live worker for model "
                    f"'{request.model_name}'",
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

        log.info(
            "pool_closing workers=%s pending=%s", len(self._workers), len(self._futures)
        )
        self._supervisor_stop.set()
        for worker in self._workers.values():
            if worker.process is None or not worker.process.is_alive():
                continue
            try:
                worker.task_queue.put(None, timeout=0.2)
            except Exception:
                pass

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
                "EmbeddedProcessPoolStrategy closed before request completed",
            )
        )
        self._stop_logging()
        with self._lock:
            self._close_complete = True
        log.info("pool_closed")

    def _launch_worker(self, worker: _WorkerState) -> None:
        worker.generation += 1
        worker.alive = False
        worker.outstanding = 0
        worker.next_restart_at = None
        worker.task_queue = self._mp_context.Queue()
        worker.lifecycle_queue = self._mp_context.Queue(maxsize=1)
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
                dict(worker.group.environment),
                worker.group.device,
                list(sys.path),
                os.getcwd(),
                self.log_queue,
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

    def _worker_registry(self, worker: _WorkerState) -> ModelRegistry:
        child_registry = ModelRegistry()
        context = {
            "group_name": worker.group.name,
            "worker_id": worker.worker_id,
            "device": worker.group.device,
        }
        for model_name in worker.model_names:
            definition = self._registry.get(model_name)
            module_dict = dict(definition.module_dict)
            module_dict["worker_context"] = dict(context)
            child_registry.add(
                ModelDefinition.model_construct(
                    **definition.model_dump(exclude={"module_dict"}),
                    module_dict=module_dict,
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
                    )

        if (
            message.get("kind") != "READY"
            or message.get("generation") != worker.generation
        ):
            log.error(
                "worker_startup_failed worker_id=%s generation=%s error=%s",
                worker.worker_id,
                worker.generation,
                message.get("error_message", "unknown error"),
            )
            raise PlatformError(
                ErrorCode.INTERNAL_ERROR,
                f"Worker '{worker.worker_id}' startup failed: "
                f"{message.get('error_message', 'unknown error')}",
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
        self._stop_logging()

    def _stop_logging(self) -> None:
        with self._lock:
            if self._logging_stopped:
                return
            self._logging_stopped = True
        try:
            if self._logging_started:
                self.listener.stop()
        finally:
            close = getattr(self.log_queue, "close", None)
            if close is not None:
                close()
            join_thread = getattr(self.log_queue, "join_thread", None)
            if join_thread is not None:
                join_thread()

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
            group_name: len(self._alive_workers_locked(group_name))
            for group_name in group_names
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

            request_id = item.get("request_id")
            assignment = (
                item.get("worker_id"),
                item.get("generation"),
            )
            if request_id is None:
                continue

            with self._lock:
                if self._assignments.get(request_id) != assignment:
                    log.debug(
                        "worker_result_ignored request_id=%s reason=stale_assignment",
                        request_id,
                    )
                    continue
                self._assignments.pop(request_id, None)
                future = self._futures.pop(request_id, None)
                worker = self._workers.get(assignment[0])
                if worker is not None and worker.outstanding > 0:
                    worker.outstanding -= 1

            if future is None or future.done():
                continue
            try:
                if item.get("ok"):
                    future.set_result(InferenceResult.model_validate(item["payload"]))
                    log.debug(
                        "worker_result_completed request_id=%s worker_id=%s generation=%s",
                        request_id,
                        assignment[0],
                        assignment[1],
                    )
                else:
                    log.warning(
                        "worker_result_failed request_id=%s worker_id=%s generation=%s "
                        "error=%s",
                        request_id,
                        assignment[0],
                        assignment[1],
                        item.get("error_message", "Unknown worker error"),
                    )
                    future.set_exception(
                        PlatformError(
                            _load_error_code(item.get("error_code")),
                            item.get(
                                "error_message",
                                "Unknown worker error",
                            ),
                        )
                    )
            except Exception as exc:
                log.error(
                    "worker_result_processing_failed request_id=%s error=%s",
                    request_id,
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
    ) -> list[Future[InferenceResult]]:
        failed: list[Future[InferenceResult]] = []
        assignment = (worker.worker_id, worker.generation)
        for request_id, owner in list(self._assignments.items()):
            if owner != assignment:
                continue
            self._assignments.pop(request_id, None)
            future = self._futures.pop(request_id, None)
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
        self._stop_logging()

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
        futures: list[Future[InferenceResult]],
        exc: Exception,
    ) -> None:
        for future in futures:
            if not future.done():
                future.set_exception(exc)


__all__ = ["EmbeddedProcessPoolStrategy"]
