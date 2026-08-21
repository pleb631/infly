"""The mutable state aggregate for a process-pool strategy."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass

from infly.core.contracts import TaskResult
from infly.runtime.strategy.worker import WorkerState


@dataclass(slots=True, frozen=True)
class PoolSnapshot:
    """An atomic view of lifecycle flags and the current worker membership."""

    accepting: bool
    closing: bool
    close_complete: bool
    workers: tuple[WorkerState, ...]


class PoolLifecycleState:
    """Own pool state and the accepting -> closing -> closed progression.

    Components coordinate through this aggregate instead of retaining separate
    worker and task mappings. All access is protected by ``lock``.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._workers: dict[str, WorkerState] = {}
        self._futures: dict[str, Future[TaskResult]] = {}
        self._assignments: dict[str, tuple[str, int]] = {}
        self._accepting = True
        self._closing = False
        self._close_complete = False
        self._cleanup_started = False
        self._cleanup_complete = threading.Event()
        # A group is deployed before it is published to ``_workers``.  Keep
        # track of that interval so terminal cleanup can wait for its worker
        # processes to be disposed as well.
        self._registrations_in_progress = 0
        self._registrations_complete = threading.Event()
        self._registrations_complete.set()
        # Unregistration removes workers from ``_workers`` before their
        # processes and queues are released. Keep that teardown visible to
        # terminal cleanup so it cannot close shared resources too early.
        self._unregistrations_in_progress = 0
        self._unregistrations_complete = threading.Event()
        self._unregistrations_complete.set()

    def begin_closing(self) -> None:
        with self.lock:
            self._accepting = False
            self._closing = True

    def claim_cleanup(self) -> bool:
        """Start exclusive shutdown cleanup, returning whether this caller won.

        A worker-failure shutdown can already have made the pool non-accepting;
        that must not prevent a later explicit ``close`` from releasing queues.
        """
        with self.lock:
            if self._close_complete or self._cleanup_started:
                return False
            self._accepting = False
            self._closing = True
            self._cleanup_started = True
            # A restart that was scheduled before close must not create a new
            # process after this cleanup has released the old resources.
            for worker in self._workers.values():
                worker.retiring = True
            return True

    def mark_closed(self) -> None:
        with self.lock:
            self.begin_closing()
            self._close_complete = True
            self._cleanup_complete.set()

    def wait_for_cleanup(self) -> None:
        """Wait until the caller performing shutdown has released all resources."""
        self._cleanup_complete.wait()

    def begin_registration(self) -> None:
        """Reserve an in-flight group registration while the pool is open.

        Callers must pair this with ``end_registration`` even when worker
        startup fails.  Keeping the reservation under the pool lock closes
        the gap between checking whether registration is allowed and starting
        unpublished worker processes.
        """
        with self.lock:
            if self._closing or self._close_complete or not self._accepting:
                raise RuntimeError("ProcessPoolStrategy is closed")
            self._registrations_in_progress += 1
            self._registrations_complete.clear()

    def end_registration(self) -> None:
        """Release an in-flight group registration reservation."""
        with self.lock:
            if self._registrations_in_progress <= 0:
                raise RuntimeError("No group registration is in progress")
            self._registrations_in_progress -= 1
            if self._registrations_in_progress == 0:
                self._registrations_complete.set()

    def wait_for_registrations(self) -> None:
        """Wait until all unpublished worker groups have been cleaned up."""
        self._registrations_complete.wait()

    def begin_unregistration(self) -> None:
        """Reserve an in-flight group teardown while the pool is open."""
        with self.lock:
            if self._closing or self._close_complete:
                raise RuntimeError("ProcessPoolStrategy is closed")
            self._unregistrations_in_progress += 1
            self._unregistrations_complete.clear()

    def end_unregistration(self) -> None:
        """Release an in-flight group teardown reservation."""
        with self.lock:
            if self._unregistrations_in_progress <= 0:
                raise RuntimeError("No group unregistration is in progress")
            self._unregistrations_in_progress -= 1
            if self._unregistrations_in_progress == 0:
                self._unregistrations_complete.set()

    def wait_for_unregistrations(self) -> None:
        """Wait until detached workers have released their resources."""
        self._unregistrations_complete.wait()

    @property
    def accepting(self) -> bool:
        with self.lock:
            return self._accepting

    @property
    def closing(self) -> bool:
        with self.lock:
            return self._closing

    @property
    def close_complete(self) -> bool:
        with self.lock:
            return self._close_complete

    @property
    def registration_closed(self) -> bool:
        with self.lock:
            return self._closing or self._close_complete or not self._accepting

    def add_workers(self, workers: list[WorkerState]) -> None:
        """Publish workers atomically once their group becomes routable."""
        with self.lock:
            self._workers.update({worker.worker_id: worker for worker in workers})

    def worker_snapshot(self) -> tuple[WorkerState, ...]:
        """Return a stable worker view for a single scheduling operation."""
        with self.lock:
            return tuple(self._workers.values())

    def snapshot(self) -> PoolSnapshot:
        """Capture lifecycle flags and workers under one lock acquisition."""
        with self.lock:
            return PoolSnapshot(
                accepting=self._accepting,
                closing=self._closing,
                close_complete=self._close_complete,
                workers=tuple(self._workers.values()),
            )

    def is_current_worker(self, worker: WorkerState) -> bool:
        with self.lock:
            return (
                not self._closing
                and not worker.retiring
                and self._workers.get(worker.worker_id) is worker
            )

    def has_task(self, task_id: str) -> bool:
        with self.lock:
            return task_id in self._futures

    def register_task(self, task_id: str, future: Future[TaskResult]) -> None:
        with self.lock:
            self._futures[task_id] = future

    def assign_task(self, task_id: str, worker: WorkerState) -> None:
        with self.lock:
            self._assignments[task_id] = (worker.worker_id, worker.generation)
            worker.outstanding += 1

    def undo_task_assignment(self, task_id: str, worker: WorkerState) -> None:
        with self.lock:
            assignment = (worker.worker_id, worker.generation)
            if self._assignments.get(task_id) == assignment:
                self._assignments.pop(task_id, None)
                worker.outstanding = max(worker.outstanding - 1, 0)

    def discard_task(self, task_id: str) -> None:
        with self.lock:
            self._futures.pop(task_id, None)
            self._assignments.pop(task_id, None)

    def take_result(
        self,
        task_id: str,
        assignment: tuple[str, int],
    ) -> tuple[bool, Future[TaskResult] | None]:
        with self.lock:
            if self._assignments.get(task_id) != assignment:
                return False, None
            self._assignments.pop(task_id, None)
            future = self._futures.pop(task_id, None)
            worker = self._workers.get(assignment[0])
            if worker is not None and worker.outstanding > 0:
                worker.outstanding -= 1
            return True, future

    def take_worker_tasks(self, worker: WorkerState) -> list[Future[TaskResult]]:
        with self.lock:
            return self._take_worker_futures(worker)

    def pending_task_count(self) -> int:
        with self.lock:
            return len(self._futures)

    def fail_tasks(self, futures: list[Future[TaskResult]], exc: Exception) -> None:
        self._fail(futures, exc)

    def fail_all_tasks(self, exc: Exception) -> None:
        with self.lock:
            futures = list(self._futures.values())
            self._futures.clear()
            self._assignments.clear()
            for worker in self._workers.values():
                worker.outstanding = 0
        self._fail(futures, exc)

    def retire_group(self, group_name: str) -> tuple[list[WorkerState], list[Future[TaskResult]]]:
        """Detach a group's workers and assigned futures.

        The caller stops returned workers and fails returned futures after
        releasing ``lock``.
        """
        with self.lock:
            workers = [worker for worker in self._workers.values() if worker.group.name == group_name]
            failed: list[Future[TaskResult]] = []
            for worker in workers:
                worker.retiring = True
                worker.alive = False
                worker.next_restart_at = None
                failed.extend(self._take_worker_futures(worker))
                self._workers.pop(worker.worker_id, None)
            return workers, failed

    def _take_worker_futures(self, worker: WorkerState) -> list[Future[TaskResult]]:
        failed: list[Future[TaskResult]] = []
        assignment = (worker.worker_id, worker.generation)
        for task_id, owner in list(self._assignments.items()):
            if owner != assignment:
                continue
            self._assignments.pop(task_id, None)
            future = self._futures.pop(task_id, None)
            if future is not None:
                failed.append(future)
        worker.outstanding = 0
        return failed

    @staticmethod
    def _fail(futures: list[Future[TaskResult]], exc: Exception) -> None:
        for future in futures:
            if not future.done():
                future.set_exception(exc)
