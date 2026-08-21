from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from concurrent.futures import Future
from contextlib import suppress
from multiprocessing import Queue, get_context

from setproctitle import setproctitle

from infly.core.contracts import TaskRequest, TaskResult
from infly.core.errors import ErrorCode, PlatformError
from infly.runtime.config import WorkerGroup
from infly.runtime.executor import HandlerExecutor
from infly.runtime.log import (
    LoggingSettings,
    MainLogManager,
    get_logger,
    setup_main_logging,
    setup_worker_logging,
)
from infly.runtime.observability import StrategyHealthSnapshot
from infly.runtime.registry import HandlerRegistry
from infly.runtime.strategy.dispatch import RequestDispatcher
from infly.runtime.strategy.groups import WorkerGroupCatalog
from infly.runtime.strategy.health import ProcessPoolHealthReporter
from infly.runtime.strategy.lifecycle import WorkerProcessController
from infly.runtime.strategy.manager import WorkerManager
from infly.runtime.strategy.results import ResultCollector
from infly.runtime.strategy.router import ProcessPoolRouter
from infly.runtime.strategy.state import PoolLifecycleState
from infly.runtime.strategy.worker import WorkerState

from .worker import restore_parent_import_path as _restore_parent_import_path
from .worker import run_worker_loop


def _worker_loop(
    worker_id: str,
    generation: int,
    task_queue: Queue,
    result_queue: Queue,
    lifecycle_queue: Queue,
    registry: HandlerRegistry,
    environment: Mapping[str, str],
    parent_sys_path: list[str],
    parent_cwd: str,
    log_queue: Queue,
    log_settings: LoggingSettings,
) -> None:
    """Compatibility adapter and multiprocessing entry point for a worker."""
    run_worker_loop(
        worker_id=worker_id,
        generation=generation,
        task_queue=task_queue,
        result_queue=result_queue,
        lifecycle_queue=lifecycle_queue,
        registry=registry,
        environment=environment,
        parent_sys_path=parent_sys_path,
        parent_cwd=parent_cwd,
        log_queue=log_queue,
        log_settings=log_settings,
        setup_logging=setup_worker_logging,
        set_process_title=setproctitle,
        restore_import_path=_restore_parent_import_path,
        executor_type=HandlerExecutor,
    )


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
        self._group_catalog = WorkerGroupCatalog(registry)
        self._state = PoolLifecycleState()
        self._mp_context = get_context("spawn")
        self._result_stop = threading.Event()
        self._supervisor_stop = threading.Event()
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
            if startup_timeout_seconds <= 0:
                raise PlatformError(
                    ErrorCode.INTERNAL_ERROR,
                    "startup_timeout_seconds must be greater than zero",
                )

            self._router = ProcessPoolRouter(self._group_catalog.groups)
            self._dispatcher = RequestDispatcher(
                catalog=self._group_catalog,
                router=self._router,
                state=self._state,
            )
            initial_workers = self._group_catalog.configure_initial(worker_groups)
            self._state.add_workers(initial_workers)

            self._result_queue: Queue = self._queue_factory()
            self._worker_controller = WorkerProcessController(
                registry=self._registry,
                mp_context=self._mp_context,
                queue_factory=self._queue_factory,
                result_queue=self._result_queue,
                log_manager=self._log_manager,
                log_settings=self._log_settings,
                worker_target=_worker_loop,
                startup_timeout_seconds=self._startup_timeout_seconds,
            )
            self._result_collector = ResultCollector(
                result_queue=self._result_queue,
                state=self._state,
            )
            self._worker_manager = WorkerManager(
                state=self._state,
                catalog=self._group_catalog,
                router=self._router,
                controller=self._worker_controller,
                startup_timeout_seconds=self._startup_timeout_seconds,
                stop_event=self._supervisor_stop,
                shutdown_pool=self._shutdown_after_worker_failure,
            )
            self._worker_manager.start_initial(self._state.worker_snapshot())
        except Exception as exc:
            self._abort_startup()
            if isinstance(exc, PlatformError):
                raise
            raise PlatformError(
                ErrorCode.INTERNAL_ERROR,
                f"Worker pool startup failed: {exc}",
            ) from exc

        self._result_thread = threading.Thread(
            target=self._result_collector.run,
            args=(self._result_stop,),
            name="EmbeddedProcessPoolResultLoop",
            daemon=True,
        )
        self._supervisor_thread = threading.Thread(
            target=self._worker_manager.run,
            name="EmbeddedProcessPoolSupervisor",
            daemon=True,
        )
        self._result_thread.start()
        self._supervisor_thread.start()
        log.info("pool_started workers=%s", len(self._state.worker_snapshot()))

    def execute(
        self,
        request: TaskRequest,
    ) -> Future[TaskResult]:
        with self._state.lock:
            return self._dispatcher.submit_locked(request)

    def register_worker_group(self, group: WorkerGroup) -> None:
        """Start and publish a new worker group without interrupting the pool.

        The group is not eligible for routing until every one of its workers has
        completed handler preloading successfully.
        """

        self._worker_manager.register(group)

    def unregister_worker_group(self, group_name: str) -> None:
        """Remove a worker group from routing and stop all of its workers.

        Requests already assigned to the group fail with ``WORKER_UNAVAILABLE``;
        no new request can be assigned once this method starts unloading it.
        """

        self._worker_manager.unregister(group_name)

    # "Unload" is an equally useful name for the lifecycle operation and keeps
    # callers from having to translate between deployment and process-pool terms.
    unload_worker_group = unregister_worker_group

    def close(self) -> None:
        self._finalize_shutdown(
            PlatformError(
                ErrorCode.INTERNAL_ERROR,
                "ProcessPoolStrategy closed before request completed",
            ),
            graceful=True,
            terminate_after=1,
        )

    @property
    def log_manager(self) -> MainLogManager:
        return self._log_manager

    def health_snapshot(self) -> StrategyHealthSnapshot:
        with self._state.lock:
            state_snapshot = self._state.snapshot()
            groups = dict(self._group_catalog.groups)
        # Checking process liveness acquires per-worker lifecycle locks.  Do
        # not keep the pool-state lock while doing that: a concurrent worker
        # launch must not block request dispatch merely because health was read.
        return ProcessPoolHealthReporter.snapshot(
            name=self.name,
            groups=groups,
            state=state_snapshot,
        )

    def _abort_startup(self) -> None:
        log.error("pool_startup_aborted")
        self._state.begin_closing()
        controller = getattr(self, "_worker_controller", None)
        if controller is not None:
            workers = self._state.worker_snapshot()
            for worker in workers:
                controller.stop(worker, terminate_after=0)
            controller.close_queues(workers)
        else:
            self._abort_startup_without_controller()
        manager = getattr(self, "_log_manager", None)
        if manager is not None:
            manager.stop()

    def _abort_startup_without_controller(self) -> None:
        """Keep startup cleanup safe for partially constructed instances."""
        for worker in self._state.worker_snapshot():
            process = worker.process
            if process is not None:
                process.join(timeout=0)
                close = getattr(process, "close", None)
                if close is not None:
                    with suppress(Exception):
                        close()
                worker.alive = False
            for queue_name in ("task_queue", "lifecycle_queue"):
                queue = getattr(worker, queue_name, None)
                WorkerProcessController.close_queue(queue)
                setattr(worker, queue_name, None)
        WorkerProcessController.close_queue(getattr(self, "_result_queue", None))

    def _shutdown_after_worker_failure(self, failed_worker: WorkerState) -> None:
        log.error(
            "pool_shutdown_after_worker_failure worker_id=%s generation=%s",
            failed_worker.worker_id,
            failed_worker.generation,
        )
        self._finalize_shutdown(
            PlatformError(
                ErrorCode.WORKER_UNAVAILABLE,
                f"Pool shut down after worker '{failed_worker.worker_id}' failed",
            ),
            graceful=False,
            terminate_after=0,
        )

    def _finalize_shutdown(
        self,
        pending_error: PlatformError,
        *,
        graceful: bool,
        terminate_after: float,
    ) -> None:
        """Perform the one, complete cleanup sequence for every terminal path."""
        if not self._state.claim_cleanup():
            self._state.wait_for_cleanup()
            log.debug("pool_close_skipped reason=cleanup_in_progress_or_complete")
            return

        try:
            workers = self._state.worker_snapshot()
            log.info("pool_closing workers=%s pending=%s", len(workers), self._state.pending_task_count())
            self._supervisor_stop.set()
            # A registration starts processes before adding their workers to
            # the state snapshot.  Let it observe the stop event and dispose
            # those unpublished processes before closing shared resources.
            self._state.wait_for_registrations()
            # An unregistration removes workers from the state snapshot before
            # it stops their processes. Wait for that teardown as well, or a
            # detached worker could outlive the shared result and log queues.
            self._state.wait_for_unregistrations()
            workers = self._state.worker_snapshot()
            if graceful:
                for worker in workers:
                    self._worker_controller.request_graceful_stop(worker)
            for worker in workers:
                self._worker_controller.stop(worker, terminate_after=terminate_after)

            self._result_stop.set()
            if hasattr(self, "_result_thread") and threading.current_thread() is not self._result_thread:
                self._result_thread.join(timeout=2)
            if hasattr(self, "_supervisor_thread") and threading.current_thread() is not self._supervisor_thread:
                self._supervisor_thread.join(timeout=2)

            self._fail_all_pending(pending_error)
            self._worker_controller.close_queues(workers)
            self._log_manager.stop()
            log.info("pool_closed")
        finally:
            self._state.mark_closed()

    def _fail_all_pending(self, exc: Exception) -> None:
        self._state.fail_all_tasks(exc)


__all__ = ["ProcessPoolStrategy"]
