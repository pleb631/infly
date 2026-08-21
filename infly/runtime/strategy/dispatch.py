"""Request-to-worker routing for the process-pool strategy."""

from __future__ import annotations

from concurrent.futures import Future

from infly.core.contracts import TaskRequest, TaskResult
from infly.core.errors import ErrorCode, PlatformError
from infly.runtime.log import get_logger
from infly.runtime.strategy.groups import WorkerGroupCatalog
from infly.runtime.strategy.router import ProcessPoolRouter
from infly.runtime.strategy.state import PoolLifecycleState

log = get_logger("infly")


class RequestDispatcher:
    """Validate and enqueue requests on a live worker.

    ``submit_locked`` must be called while the process-pool state lock is held.
    """

    def __init__(
        self,
        *,
        catalog: WorkerGroupCatalog,
        router: ProcessPoolRouter,
        state: PoolLifecycleState,
    ) -> None:
        self._catalog = catalog
        self._router = router
        self._state = state

    def submit_locked(self, request: TaskRequest) -> Future[TaskResult]:
        future: Future[TaskResult] = Future()
        if not request.task_id:
            future.set_exception(
                PlatformError(
                    ErrorCode.INVALID_REQUEST,
                    "TaskRequest must be submitted through TaskScheduler before execution.",
                )
            )
            return future
        if not self._state.accepting:
            log.warning("request_rejected task_id=%s reason=pool_closed", request.task_id)
            future.set_exception(PlatformError(ErrorCode.INTERNAL_ERROR, "ProcessPoolStrategy is closed"))
            return future
        if self._state.has_task(request.task_id):
            log.warning("request_rejected task_id=%s reason=duplicate", request.task_id)
            future.set_exception(PlatformError(ErrorCode.INTERNAL_ERROR, f"Duplicate task_id: {request.task_id}"))
            return future

        workers = self._state.worker_snapshot()
        live_workers_by_group = self._router.live_workers_by_group(workers)
        group_names = [
            group_name
            for group_name in self._catalog.handler_groups.get(request.handler_name, [])
            if live_workers_by_group.get(group_name)
        ]
        if not group_names:
            log.warning(
                "request_rejected task_id=%s handler=%s reason=no_live_worker",
                request.task_id,
                request.handler_name,
            )
            future.set_exception(
                PlatformError(
                    ErrorCode.WORKER_UNAVAILABLE,
                    f"No live worker is deployed for handler '{request.handler_name}'",
                )
            )
            return future

        self._state.register_task(request.task_id, future)
        remaining_groups = list(group_names)
        while remaining_groups:
            group_name = self._router.select_group(remaining_groups, live_workers_by_group)
            for worker in self._router.ordered_workers(group_name, live_workers_by_group):
                self._state.assign_task(request.task_id, worker)
                try:
                    worker.task_queue.put_nowait(request)
                except Exception as exc:
                    self._state.undo_task_assignment(request.task_id, worker)
                    log.warning(
                        "request_assignment_failed task_id=%s worker_id=%s generation=%s error=%s",
                        request.task_id,
                        worker.worker_id,
                        worker.generation,
                        exc,
                        exc_info=True,
                    )
                    continue
                self._router.record_assignment(worker)
                log.debug(
                    "request_assigned task_id=%s handler=%s worker_id=%s generation=%s",
                    request.task_id,
                    request.handler_name,
                    worker.worker_id,
                    worker.generation,
                )
                return future
            remaining_groups.remove(group_name)

        self._state.discard_task(request.task_id)
        log.warning(
            "request_rejected task_id=%s handler=%s reason=assignment_failed",
            request.task_id,
            request.handler_name,
        )
        future.set_exception(
            PlatformError(
                ErrorCode.WORKER_UNAVAILABLE,
                f"Unable to submit request to a live worker for handler '{request.handler_name}'",
            )
        )
        return future
