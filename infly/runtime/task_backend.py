from __future__ import annotations

import copy as _copy
import datetime
import heapq
import threading
import time

from infly.core.contracts import (
    TaskRecord,
    TaskResult,
    TaskStatus,
)
from infly.core.errors import ErrorCode
from infly.core.ports import TaskBackend
from infly.runtime.log import get_logger

log = get_logger()


class InMemoryTaskBackend:
    def __init__(self, *, max_retained_terminal_tasks: int = 50) -> None:
        if max_retained_terminal_tasks < 0:
            raise ValueError("max_retained_terminal_tasks must be non-negative")
        self._max_retained_terminal_tasks = max_retained_terminal_tasks
        self._records: dict[str, TaskRecord] = {}
        self._pending: list[tuple[int, int, str]] = []
        self._terminal_finished_at: dict[str, tuple[int, int]] = {}
        self._read_terminal_tasks: set[str] = set()
        self._terminal_sequence = 0
        self._sequence = 0
        self._lock = threading.Lock()

    def submit(self, record: TaskRecord, priority: int = 0) -> None:
        with self._lock:
            if record.task_id in self._records:
                log.warning(
                    "task_submit_rejected task_id=%s reason=duplicate", record.task_id
                )
                raise ValueError(f"Task '{record.task_id}' already exists.")
            self._records[record.task_id] = record
            try:
                heapq.heappush(
                    self._pending,
                    (-priority, self._sequence, record.task_id),
                )
            except Exception:
                del self._records[record.task_id]
                raise
            self._sequence += 1
        log.info(
            "task_submitted task_id=%s task_key=%s handler=%s priority=%s",
            record.task_id,
            record.request.task_key,
            record.request.handler_name,
            priority,
        )

    def pull(self) -> str | None:
        with self._lock:
            if not self._pending:
                return None
            _, _, task_id = heapq.heappop(self._pending)
        log.info("task_pulled task_id=%s", task_id)
        return task_id

    def get(self, task_id: str, copy: bool = False) -> TaskRecord | None:
        with self._lock:
            if copy:
                record = self._records.get(task_id)
                if record is None:
                    return None
                return _copy.deepcopy(record)
            return self._records.get(task_id)

    def read(
        self,
        task_id: str,
        *,
        consume: bool = False,
    ) -> TaskRecord | None:
        with self._lock:
            record = self._records.get(task_id)
            if record is None or record.status not in {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
            }:
                return None
            if consume:
                self._terminal_finished_at.pop(task_id, None)
                self._read_terminal_tasks.discard(task_id)
                return self._records.pop(task_id)
            self._read_terminal_tasks.add(task_id)
            return _copy.deepcopy(record)

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: TaskResult | None = None,
        error_code: ErrorCode | None = None,
        error_message: str | None = None,
    ) -> TaskRecord:
        with self._lock:
            record = self._records.get(task_id)
            if record is None:
                raise KeyError(task_id)
            updated = TaskRecord(
                task_id=record.task_id,
                request=record.request,
                status=status,
                result=result,
                error_code=error_code,
                error_message=error_message,
                created_at=record.created_at,
                updated_at=datetime.datetime.now(datetime.UTC),
            )
            self._records[task_id] = updated
            if status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
                if task_id not in self._terminal_finished_at:
                    self._terminal_finished_at[task_id] = (
                        time.monotonic_ns(),
                        self._terminal_sequence,
                    )
                    self._terminal_sequence += 1
                self._prune_terminal_records_locked()
            else:
                self._terminal_finished_at.pop(task_id, None)
                self._read_terminal_tasks.discard(task_id)
        log.info(
            "task_status_updated task_id=%s status=%s error_code=%s",
            task_id,
            status,
            error_code,
        )
        return updated

    def _prune_terminal_records_locked(self) -> None:
        limit = self._max_retained_terminal_tasks
        if limit == 0:
            return
        terminal_task_ids = sorted(
            (
                task_id
                for task_id, record in self._records.items()
                if record.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}
            ),
            key=lambda task_id: (
                0 if task_id in self._read_terminal_tasks else 1,
                self._terminal_finished_at[task_id],
            ),
        )
        for task_id in terminal_task_ids[:-limit]:
            record = self._records.pop(task_id)
            self._terminal_finished_at.pop(task_id, None)
            self._read_terminal_tasks.discard(task_id)
            log.debug(
                "terminal_task_evicted task_id=%s finished_at=%s",
                task_id,
                record.updated_at.isoformat(),
            )

    def list_all(self) -> list[TaskRecord]:
        with self._lock:
            return [_copy.deepcopy(record) for record in self._records.values()]


__all__ = ["InMemoryTaskBackend", "TaskBackend"]
