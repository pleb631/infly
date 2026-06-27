import threading
import time
from concurrent.futures import Future

import pytest

from infly.core.contracts import (
    TaskRequest,
    TaskResult,
    TaskRecord,
    TaskStatus,
)
from infly.core.errors import ErrorCode, PlatformError
from infly.runtime.scheduler import TaskScheduler
from infly.runtime.task_backend import InMemoryTaskBackend
from infly.runtime.config import SchedulerConfig


def _request(task_key: str = "req-1") -> TaskRequest:
    return TaskRequest(
        task_key=task_key,
        handler_name="echo",
        input={"text": "ok"},
        caller="test",
    )


def _wait_for_status(
    scheduler: TaskScheduler, task_id: str, *statuses: TaskStatus
) -> object:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        response = scheduler.query(task_id)
        if response.status in statuses:
            return response
        time.sleep(0.01)
    return scheduler.query(task_id)


def _result(request: TaskRequest) -> TaskResult:
    return TaskResult(
        task_key=request.task_key,
        output={"echo": request.input["text"]},
    )


class SuccessStrategy:
    def execute(self, request: TaskRequest) -> Future[TaskResult]:
        future: Future[TaskResult] = Future()
        future.set_result(_result(request))
        return future

    def close(self) -> None:
        pass


class BlockingStrategy:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()

    def execute(self, request: TaskRequest) -> Future[TaskResult]:
        future: Future[TaskResult] = Future()
        self.started.set()

        def complete() -> None:
            self.release.wait()
            future.set_result(_result(request))

        threading.Thread(target=complete, daemon=True).start()
        return future

    def close(self) -> None:
        self.closed.set()


class WorkerUnavailableStrategy:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, request: TaskRequest) -> Future[TaskResult]:
        self.calls.append(request.task_key)
        future: Future[TaskResult] = Future()
        future.set_exception(
            PlatformError(ErrorCode.WORKER_UNAVAILABLE, "worker exited")
        )
        return future

    def close(self) -> None:
        pass


def _scheduler(
    strategy: object, *, max_outstanding_tasks: int = 8
) -> TaskScheduler:
    return TaskScheduler(
        strategy,  # type: ignore[arg-type]
        scheduler_config=SchedulerConfig(
            max_outstanding_tasks=max_outstanding_tasks,
            num_threads=1,
        ),
    )


def test_scheduler_completes_submitted_task() -> None:
    scheduler = _scheduler(SuccessStrategy())
    scheduler.start()
    try:
        task_id = scheduler.submit(_request())
        response = _wait_for_status(scheduler, task_id, TaskStatus.COMPLETED)
    finally:
        scheduler.stop()

    assert response.status == TaskStatus.COMPLETED
    assert response.result is not None
    assert response.result.output == {"echo": "ok"}


def test_terminal_query_retains_record_by_default() -> None:
    scheduler = _scheduler(SuccessStrategy())
    scheduler.start()
    try:
        task_id = scheduler.submit(_request())
        first = scheduler.query(task_id, wait=True)
        second = scheduler.query(task_id)
    finally:
        scheduler.stop()

    assert first.status == TaskStatus.COMPLETED
    assert second == first
    assert scheduler.backend.get(task_id) is not None


def test_terminal_query_returns_isolated_copy() -> None:
    scheduler = _scheduler(SuccessStrategy())
    scheduler.start()
    try:
        task_id = scheduler.submit(_request())
        first = scheduler.query(task_id, wait=True)
        assert first.result is not None
        first.result.output["mutated"] = True

        second = scheduler.query(task_id)
    finally:
        scheduler.stop()

    assert second.result is not None
    assert "mutated" not in second.result.output
    assert scheduler.backend.get(task_id).result is not None
    assert "mutated" not in scheduler.backend.get(task_id).result.output  # type: ignore[union-attr]


def test_terminal_query_can_consume_record() -> None:
    scheduler = _scheduler(SuccessStrategy())
    scheduler.start()
    try:
        task_id = scheduler.submit(_request())
        response = scheduler.query(task_id, wait=True, consume=True)

        with pytest.raises(PlatformError) as caught:
            scheduler.query(task_id)
    finally:
        scheduler.stop()

    assert response.status == TaskStatus.COMPLETED
    assert caught.value.code == ErrorCode.NOT_FOUND


def test_terminal_query_can_consume_record_without_wait() -> None:
    scheduler = _scheduler(SuccessStrategy())
    scheduler.start()
    try:
        task_id = scheduler.submit(_request())
        _wait_for_status(scheduler, task_id, TaskStatus.COMPLETED)
        response = scheduler.query(task_id, consume=True)

        with pytest.raises(PlatformError) as caught:
            scheduler.query(task_id)
    finally:
        scheduler.stop()

    assert response.status == TaskStatus.COMPLETED
    assert caught.value.code == ErrorCode.NOT_FOUND


def test_default_backend_uses_scheduler_terminal_retention_limit() -> None:
    scheduler = TaskScheduler(
        SuccessStrategy(),
        scheduler_config=SchedulerConfig(
            max_outstanding_tasks=2,
            num_threads=1,
            max_retained_terminal_tasks=1,
        ),
    )
    scheduler.start()
    try:
        first_id = scheduler.submit(_request("first"))
        second_id = scheduler.submit(_request("second"))
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            second = scheduler.backend.get(second_id)
            if second is not None and second.status == TaskStatus.COMPLETED:
                break
            time.sleep(0.01)
    finally:
        scheduler.stop()

    assert scheduler.backend.get(first_id) is None
    assert scheduler.backend.get(second_id) is not None


def test_scheduler_stop_closes_strategy() -> None:
    strategy = BlockingStrategy()
    scheduler = _scheduler(strategy)

    scheduler.start()
    try:
        scheduler.submit(_request())
        assert strategy.started.wait(1)
    finally:
        strategy.release.set()
        scheduler.stop()

    assert strategy.closed.is_set()


def test_scheduler_stop_is_idempotent_before_start() -> None:
    strategy = SuccessStrategy()
    scheduler = _scheduler(strategy)

    scheduler.stop()
    scheduler.stop()


def test_scheduler_start_is_idempotent_when_called_twice() -> None:
    scheduler = _scheduler(SuccessStrategy())

    scheduler.start()
    scheduler.start()
    try:
        assert len(scheduler._threads) == 1
    finally:
        scheduler.stop()


def test_scheduler_stop_fails_pending_tasks_and_releases_slots() -> None:
    scheduler = _scheduler(SuccessStrategy(), max_outstanding_tasks=1)
    task_id = scheduler.submit(_request("pending"))

    scheduler.stop()

    response = scheduler.query(task_id)
    assert response.status == TaskStatus.FAILED
    assert response.error_code == ErrorCode.WORKER_UNAVAILABLE

    with pytest.raises(PlatformError) as caught:
        scheduler.submit(_request("replacement"))

    assert caught.value.code == ErrorCode.INVALID_STATE


def test_scheduler_start_is_rejected_after_stop() -> None:
    scheduler = _scheduler(SuccessStrategy())

    scheduler.stop()

    with pytest.raises(PlatformError) as caught:
        scheduler.start()

    assert caught.value.code == ErrorCode.INVALID_STATE


def test_query_wait_returns_failed_after_stop() -> None:
    scheduler = _scheduler(SuccessStrategy(), max_outstanding_tasks=1)
    task_id = scheduler.submit(_request("pending"))
    result: dict[str, object] = {}
    errors: list[BaseException] = []

    def wait_for_terminal() -> None:
        try:
            result["response"] = scheduler.query(task_id, wait=True)
        except BaseException as exc:  # pragma: no cover - defensive
            errors.append(exc)

    thread = threading.Thread(target=wait_for_terminal)
    thread.start()
    scheduler.stop()
    thread.join(1)

    assert errors == []
    response = result["response"]
    assert response.status == TaskStatus.FAILED
    assert response.error_code == ErrorCode.WORKER_UNAVAILABLE


def test_scheduler_stop_ignores_foreign_pending_backend_records() -> None:
    backend = InMemoryTaskBackend()
    foreign = TaskRecord(task_id="foreign", request=_request("foreign"))
    backend.submit(foreign)
    scheduler = TaskScheduler(
        SuccessStrategy(),
        backend=backend,
        scheduler_config=SchedulerConfig(max_outstanding_tasks=1),
    )

    scheduler.stop()

    assert scheduler.query("foreign").status == TaskStatus.PENDING


def test_scheduler_stop_retains_threads_that_miss_join_timeout() -> None:
    release = threading.Event()
    started = threading.Event()

    class StuckStrategy:
        def execute(self, request: TaskRequest) -> Future[TaskResult]:
            started.set()
            release.wait()
            future: Future[TaskResult] = Future()
            future.set_result(_result(request))
            return future

        def close(self) -> None:
            pass

    scheduler = TaskScheduler(
        StuckStrategy(),
        scheduler_config=SchedulerConfig(num_threads=1),
    )
    scheduler.start()
    task_id = scheduler.submit(_request("stuck"))
    try:
        assert started.wait(1)
        assert scheduler.query(task_id).status == TaskStatus.RUNNING
        scheduler.stop(timeout=0)
        assert len(scheduler._threads) == 1
        assert scheduler._threads[0].is_alive()
    finally:
        release.set()
        scheduler.stop(timeout=1)

    assert scheduler._threads == []


def test_scheduler_last_worker_exit_fails_tasks_left_pending_after_stop_timeout() -> None:
    release = threading.Event()
    started = threading.Event()

    class StuckStrategy:
        def execute(self, request: TaskRequest) -> Future[TaskResult]:
            started.set()
            release.wait()
            future: Future[TaskResult] = Future()
            future.set_result(_result(request))
            return future

        def close(self) -> None:
            pass

    scheduler = TaskScheduler(
        StuckStrategy(),
        scheduler_config=SchedulerConfig(
            max_outstanding_tasks=2,
            num_threads=1,
        ),
    )
    scheduler.start()
    first_task_id = scheduler.submit(_request("running"))
    assert started.wait(1)
    pending_task_id = scheduler.submit(_request("pending"))

    scheduler.stop(timeout=0)
    release.set()

    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and scheduler._threads:
        time.sleep(0.01)

    assert scheduler.query(first_task_id).status == TaskStatus.COMPLETED
    pending = scheduler.query(pending_task_id)
    assert pending.status == TaskStatus.FAILED
    assert pending.error_code == ErrorCode.WORKER_UNAVAILABLE
    assert scheduler._threads == []


def test_scheduler_stop_rejects_negative_timeout() -> None:
    scheduler = _scheduler(SuccessStrategy())

    with pytest.raises(PlatformError) as caught:
        scheduler.stop(timeout=-1)

    assert caught.value.code == ErrorCode.INVALID_ARGUMENT


def test_query_without_wait_returns_current_non_terminal_status() -> None:
    strategy = BlockingStrategy()
    scheduler = _scheduler(strategy)
    pending_task_id = scheduler.submit(_request("pending"))

    assert scheduler.query(pending_task_id).status == TaskStatus.PENDING

    scheduler.start()
    try:
        assert strategy.started.wait(1)
        assert scheduler.query(pending_task_id).status == TaskStatus.RUNNING
    finally:
        strategy.release.set()
        scheduler.stop()


def test_pending_tasks_consume_outstanding_slots() -> None:
    scheduler = _scheduler(SuccessStrategy(), max_outstanding_tasks=1)
    accepted_task_id = scheduler.submit(_request("accepted"))

    with pytest.raises(PlatformError) as caught:
        scheduler.submit(_request("rejected"))

    assert caught.value.code == ErrorCode.OVERLOADED
    assert [record.task_id for record in scheduler.backend.list_all()] == [
        accepted_task_id
    ]


def test_submit_accepts_priority_as_keyword_argument() -> None:
    scheduler = _scheduler(SuccessStrategy())
    low_task_id = scheduler.submit(_request("low"), priority=0)
    high_task_id = scheduler.submit(_request("high"), priority=10)

    assert scheduler.backend.pull() == high_task_id
    assert scheduler.backend.pull() == low_task_id


def test_running_tasks_consume_outstanding_slots() -> None:
    strategy = BlockingStrategy()
    scheduler = _scheduler(strategy, max_outstanding_tasks=1)
    scheduler.start()
    try:
        accepted_task_id = scheduler.submit(_request("running"))
        assert strategy.started.wait(1)
        assert scheduler.query(accepted_task_id).status == TaskStatus.RUNNING

        with pytest.raises(PlatformError) as caught:
            scheduler.submit(_request("rejected"))
    finally:
        strategy.release.set()
        scheduler.stop()

    assert caught.value.code == ErrorCode.OVERLOADED
    assert len(scheduler.backend.list_all()) == 1


def test_concurrent_submissions_atomically_enforce_outstanding_limit() -> None:
    limit = 4
    scheduler = _scheduler(SuccessStrategy(), max_outstanding_tasks=limit)
    barrier = threading.Barrier(20)
    accepted: list[str] = []
    rejected: list[ErrorCode] = []
    lock = threading.Lock()

    def submit(index: int) -> None:
        barrier.wait()
        try:
            task_id = scheduler.submit(_request(f"req-{index}"))
        except PlatformError as exc:
            with lock:
                rejected.append(exc.code)
        else:
            with lock:
                accepted.append(task_id)

    threads = [threading.Thread(target=submit, args=(index,)) for index in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(accepted) == limit
    assert rejected == [ErrorCode.OVERLOADED] * (20 - limit)
    assert {record.task_id for record in scheduler.backend.list_all()} == set(accepted)


def test_submission_failure_rolls_back_record_and_releases_slot() -> None:
    class FailOnceBackend(InMemoryTaskBackend):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        def submit(self, record, priority: int = 0) -> None:
            if not self.failed:
                self.failed = True
                raise RuntimeError("backend failure")
            super().submit(record, priority)

    backend = FailOnceBackend()
    scheduler = TaskScheduler(
        SuccessStrategy(),
        backend=backend,
        scheduler_config=SchedulerConfig(max_outstanding_tasks=1),
    )

    with pytest.raises(RuntimeError, match="backend failure"):
        scheduler.submit(_request("failed"))

    assert backend.list_all() == []
    accepted_task_id = scheduler.submit(_request("accepted"))
    assert scheduler.query(accepted_task_id).status == TaskStatus.PENDING


@pytest.mark.parametrize(
    "strategy_factory, expected_status, expected_error_code",
    [
        (BlockingStrategy, TaskStatus.COMPLETED, None),
        (
            WorkerUnavailableStrategy,
            TaskStatus.FAILED,
            ErrorCode.WORKER_UNAVAILABLE,
        ),
    ],
)
def test_terminal_tasks_release_slots_for_later_submissions(
    strategy_factory,
    expected_status: TaskStatus,
    expected_error_code: ErrorCode | None,
) -> None:
    strategy = strategy_factory()
    scheduler = _scheduler(strategy, max_outstanding_tasks=1)
    scheduler.start()
    try:
        first_task_id = scheduler.submit(_request("first"))
        if isinstance(strategy, BlockingStrategy):
            assert strategy.started.wait(1)
            strategy.release.set()
        response = _wait_for_status(scheduler, first_task_id, expected_status)
        second_task_id = scheduler.submit(_request("second"))
    finally:
        if isinstance(strategy, BlockingStrategy):
            strategy.release.set()
        scheduler.stop()

    assert response.status == expected_status
    if expected_error_code is not None:
        assert response.error_code == expected_error_code
    if isinstance(strategy, WorkerUnavailableStrategy):
        assert strategy.calls.count("first") == 1
    second = scheduler.query(second_task_id)
    assert second.status == TaskStatus.FAILED
    assert second.error_code == ErrorCode.WORKER_UNAVAILABLE


def test_query_wait_returns_failed_status_instead_of_future_exception() -> None:
    scheduler = _scheduler(WorkerUnavailableStrategy())
    scheduler.start()
    try:
        task_id = scheduler.submit(_request())
        response = scheduler.query(task_id, wait=True)
    finally:
        scheduler.stop()

    assert response.status == TaskStatus.FAILED
    assert response.error_code == ErrorCode.WORKER_UNAVAILABLE
    assert scheduler.backend.get(task_id) is not None


def test_query_wait_times_out_when_task_does_not_complete() -> None:
    strategy = BlockingStrategy()
    scheduler = _scheduler(strategy)
    scheduler.start()
    try:
        task_id = scheduler.submit(_request())
        assert strategy.started.wait(1)

        with pytest.raises(PlatformError) as caught:
            scheduler.query(task_id, wait=True, timeout_seconds=0.05)
    finally:
        strategy.release.set()
        scheduler.stop()

    assert caught.value.code == ErrorCode.TIMEOUT
    assert "timed out" in str(caught.value).lower()


def test_query_wait_raises_not_found_when_task_disappears_during_wait() -> None:
    class DisappearingBackend(InMemoryTaskBackend):
        def __init__(self) -> None:
            super().__init__()
            self._get_counts: dict[str, int] = {}

        def get(self, task_id: str, copy: bool = False):
            count = self._get_counts.get(task_id, 0)
            self._get_counts[task_id] = count + 1
            if count >= 1:
                with self._lock:
                    self._records.pop(task_id, None)
                return None
            return super().get(task_id, copy=copy)

    scheduler = TaskScheduler(
        SuccessStrategy(),
        backend=DisappearingBackend(),
        scheduler_config=SchedulerConfig(max_outstanding_tasks=8),
    )
    task_id = scheduler.submit(_request("disappearing"))

    with pytest.raises(PlatformError) as caught:
        scheduler.query(task_id, wait=True)

    assert caught.value.code == ErrorCode.NOT_FOUND


def test_unexpected_execution_error_is_internal_and_releases_slot() -> None:
    class BrokenStrategy:
        def execute(self, request) -> Future[TaskResult]:
            raise RuntimeError("broken callback")

        def close(self) -> None:
            pass

    scheduler = _scheduler(BrokenStrategy(), max_outstanding_tasks=1)
    scheduler.start()
    try:
        first_task_id = scheduler.submit(_request("first"))
        response = _wait_for_status(scheduler, first_task_id, TaskStatus.FAILED)
        second_task_id = scheduler.submit(_request("second"))
    finally:
        scheduler.stop()

    assert response.error_code == ErrorCode.INTERNAL_ERROR
    second = scheduler.query(second_task_id)
    assert second.status == TaskStatus.FAILED
    assert second.error_code == ErrorCode.WORKER_UNAVAILABLE


def test_worker_loop_survives_transient_pull_failure() -> None:
    class PullFailsOnceBackend(InMemoryTaskBackend):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        def pull(self) -> str | None:
            if not self.failed:
                self.failed = True
                raise RuntimeError("transient pull failure")
            return super().pull()

    scheduler = TaskScheduler(
        SuccessStrategy(),
        backend=PullFailsOnceBackend(),
        scheduler_config=SchedulerConfig(num_threads=1),
    )
    scheduler.start()
    try:
        task_id = scheduler.submit(_request("survives-pull-failure"))
        response = _wait_for_status(scheduler, task_id, TaskStatus.COMPLETED)
    finally:
        scheduler.stop()

    assert response.status == TaskStatus.COMPLETED


def test_submit_and_wait_raises_execution_failure() -> None:
    scheduler = _scheduler(WorkerUnavailableStrategy())
    scheduler.start()
    try:
        with pytest.raises(PlatformError) as caught:
            scheduler.submit_and_wait(_request())
    finally:
        scheduler.stop()

    assert caught.value.code == ErrorCode.WORKER_UNAVAILABLE
    assert len(scheduler.backend.list_all()) == 1


def test_submit_and_wait_can_consume_failed_result() -> None:
    scheduler = _scheduler(WorkerUnavailableStrategy())
    scheduler.start()
    try:
        with pytest.raises(PlatformError) as caught:
            scheduler.submit_and_wait(_request(), consume=True)
    finally:
        scheduler.stop()

    assert caught.value.code == ErrorCode.WORKER_UNAVAILABLE
    assert scheduler.backend.list_all() == []


def test_submit_and_wait_times_out_with_api_context() -> None:
    strategy = BlockingStrategy()
    scheduler = _scheduler(strategy)
    scheduler.start()
    try:
        with pytest.raises(PlatformError) as caught:
            scheduler.submit_and_wait(_request(), timeout_seconds=0.05)
    finally:
        strategy.release.set()
        scheduler.stop()

    assert caught.value.code == ErrorCode.TIMEOUT
    assert "submit_and_wait" in str(caught.value)
