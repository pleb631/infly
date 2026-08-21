"""Result-queue consumption for the process-pool strategy."""

from __future__ import annotations

from queue import Empty
from typing import Any

from infly.core.errors import PlatformError
from infly.runtime.log import get_logger
from infly.runtime.strategy.state import PoolLifecycleState
from infly.runtime.strategy.worker import load_error_code

log = get_logger("infly")


class ResultCollector:
    """Match worker results to their assigned futures.

    The tracker owns the pending state. The collector only consumes the result
    queue and resolves the corresponding future.
    """

    def __init__(
        self,
        *,
        result_queue: Any,
        state: PoolLifecycleState,
    ) -> None:
        self._result_queue = result_queue
        self._state = state

    def run(self, stop_event: Any) -> None:
        while not stop_event.is_set():
            try:
                item = self._result_queue.get(timeout=0.1)
            except Empty:
                continue
            self._complete(item)

    def _complete(self, item: Any) -> None:
        task_id = item.task_id
        assignment = (item.worker_id, item.generation)
        current, future = self._state.take_result(task_id, assignment)
        if not current:
            log.debug("worker_result_ignored task_id=%s reason=stale_assignment", task_id)
            return

        if future is None or future.done():
            return
        try:
            if item.ok:
                future.set_result(item.payload)
                log.debug(
                    "worker_result_completed task_id=%s worker_id=%s generation=%s",
                    task_id,
                    assignment[0],
                    assignment[1],
                )
            else:
                message = item.error_message or "Unknown worker error"
                log.warning(
                    "worker_result_failed task_id=%s worker_id=%s generation=%s error=%s",
                    task_id,
                    assignment[0],
                    assignment[1],
                    message,
                )
                future.set_exception(PlatformError(load_error_code(item.error_code), message))
        except Exception as exc:
            log.error("worker_result_processing_failed task_id=%s error=%s", task_id, exc, exc_info=True)
            if not future.done():
                future.set_exception(exc)
