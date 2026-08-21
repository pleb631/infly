"""Worker-group lifecycle management and process supervision."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable

from infly.core.errors import ErrorCode, PlatformError
from infly.runtime.config import WorkerGroup
from infly.runtime.log import get_logger
from infly.runtime.strategy.groups import WorkerGroupCatalog
from infly.runtime.strategy.lifecycle import WorkerProcessController
from infly.runtime.strategy.router import ProcessPoolRouter
from infly.runtime.strategy.state import PoolLifecycleState
from infly.runtime.strategy.worker import WorkerState

log = get_logger("infly")


class WorkerManager:
    """Own worker deployment, removal, and failure recovery.

    ``PoolLifecycleState`` owns the worker registry and synchronization;
    ``WorkerProcessController`` owns process mechanics. This class owns the
    policy that coordinates those two collaborators.
    """

    def __init__(
        self,
        *,
        state: PoolLifecycleState,
        catalog: WorkerGroupCatalog,
        router: ProcessPoolRouter,
        controller: WorkerProcessController,
        startup_timeout_seconds: float,
        stop_event: threading.Event,
        shutdown_pool: Callable[[WorkerState], None],
    ) -> None:
        self._state = state
        self._catalog = catalog
        self._router = router
        self._controller = controller
        self._startup_timeout_seconds = startup_timeout_seconds
        self._stop_event = stop_event
        self._shutdown_pool = shutdown_pool

    def start_initial(self, workers: Iterable[WorkerState], *, interruptible: bool = False) -> None:
        workers = list(workers)
        for worker in workers:
            self._controller.launch(worker)
        if not interruptible:
            self._controller.await_initial_startup(workers)
            return
        deadline = time.monotonic() + self._startup_timeout_seconds
        for worker in workers:
            self._controller.await_ready(worker, deadline, stop_event=self._stop_event)

    def register(self, group: WorkerGroup) -> None:
        handler_names = self._catalog.resolve_handlers(group)
        workers = self._catalog.create_workers(group, handler_names)
        with self._state.lock:
            self._catalog.ensure_can_register(group.name, closed=self._state.registration_closed)
            try:
                self._state.begin_registration()
            except RuntimeError as exc:
                raise PlatformError(ErrorCode.INTERNAL_ERROR, "ProcessPoolStrategy is closed") from exc

        try:
            try:
                self.start_initial(workers, interruptible=True)
            except Exception as exc:
                self._controller.dispose(workers)
                raise PlatformError(
                    ErrorCode.INTERNAL_ERROR,
                    f"WorkerGroup '{group.name}' startup failed: {exc}",
                ) from exc

            try:
                with self._state.lock:
                    self._catalog.ensure_can_register(group.name, closed=self._state.registration_closed)
                    self._catalog.publish(group, handler_names)
                    self._state.add_workers(workers)
            except Exception:
                self._controller.dispose(workers)
                raise
            log.info("worker_group_registered group=%s workers=%s handlers=%s", group.name, len(workers), handler_names)
        finally:
            self._state.end_registration()

    def unregister(self, group_name: str) -> None:
        with self._state.lock:
            try:
                self._state.begin_unregistration()
            except RuntimeError as exc:
                raise PlatformError(ErrorCode.INTERNAL_ERROR, "ProcessPoolStrategy is closed") from exc
            try:
                self._catalog.unpublish(group_name)
                self._router.forget_group(group_name)
                workers, failed = self._state.retire_group(group_name)
            except Exception:
                self._state.end_unregistration()
                raise

        try:
            self._controller.dispose(workers)
        finally:
            # The barrier covers only process/queue cleanup. Failing futures
            # afterwards lets a callback safely call ``close`` without waiting
            # on its own unregistration operation.
            self._state.end_unregistration()
        self._state.fail_tasks(
            failed,
            PlatformError(ErrorCode.WORKER_UNAVAILABLE, f"WorkerGroup '{group_name}' was unloaded"),
        )
        log.info(
            "worker_group_unregistered group=%s workers=%s pending_failed=%s",
            group_name,
            len(workers),
            len(failed),
        )

    def run(self) -> None:
        log.info("pool_supervisor_started")
        while not self._stop_event.wait(0.05):
            now = time.monotonic()
            with self._state.lock:
                if self._state.closing:
                    log.info("pool_supervisor_stopped")
                    return
                workers = self._state.worker_snapshot()
            for worker in workers:
                with self._state.lock:
                    if not self._state.is_current_worker(worker):
                        continue
                    worker_exited = worker.alive and not worker.is_running()
                    restart_due = (
                        not worker.alive and worker.next_restart_at is not None and now >= worker.next_restart_at
                    )
                if worker_exited:
                    self._handle_exit(worker, now)
                elif restart_due:
                    self._restart(worker)

    def _handle_exit(self, worker: WorkerState, now: float) -> None:
        with self._state.lock:
            if not worker.alive or not self._state.is_current_worker(worker):
                return
            worker.alive = False
            self._router.reset_group_weight(worker.group.name)
            failed = self._state.take_worker_tasks(worker)

        self._controller.cleanup_failed_start(worker)

        log.warning(
            "worker_exited worker_id=%s generation=%s pending_failed=%s mode=%s",
            worker.worker_id,
            worker.generation,
            len(failed),
            worker.group.safety.mode,
        )
        self._state.fail_tasks(
            failed,
            PlatformError(ErrorCode.WORKER_UNAVAILABLE, f"Worker '{worker.worker_id}' exited unexpectedly"),
        )
        if worker.group.safety.mode == "shutdown":
            self._shutdown_pool(worker)
        elif worker.group.safety.mode == "restart":
            self._schedule_restart(worker, now)

    def _restart(self, worker: WorkerState) -> None:
        try:
            with self._state.lock:
                if not self._state.is_current_worker(worker):
                    return
                worker.next_restart_at = None
                worker.restart_times.append(time.monotonic())

            # ``claim_cleanup`` and ``retire_group`` set ``retiring`` before
            # trying to acquire this lock. That makes the check and launch
            # atomic with respect to teardown without nesting the lifecycle and
            # pool-state locks in the opposite order used by request routing.
            with worker.lifecycle_lock:
                if worker.retiring:
                    return
                self._controller.launch(worker)
            log.info("worker_restart_started worker_id=%s generation=%s", worker.worker_id, worker.generation)
            self._controller.await_ready(
                worker,
                time.monotonic() + self._startup_timeout_seconds,
                stop_event=self._stop_event,
            )
            with self._state.lock:
                self._router.reset_group_weight(worker.group.name)
            log.info("worker_restart_completed worker_id=%s generation=%s", worker.worker_id, worker.generation)
        except Exception as exc:
            log.error("worker_restart_failed worker_id=%s generation=%s error=%s", worker.worker_id, worker.generation, exc, exc_info=True)
            self._controller.cleanup_failed_start(worker)
            with self._state.lock:
                if not self._state.closing and not worker.retiring:
                    self._schedule_restart(worker, time.monotonic())

    def _schedule_restart(self, worker: WorkerState, now: float) -> None:
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
