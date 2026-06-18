import threading

import pytest

from infly.core.contracts import InferenceRequest, InferenceResult, TaskRecord, TaskStatus
from infly.core.errors import ErrorCode
from infly.runtime.task_backend import InMemoryTaskBackend


def _record(task_id: str) -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        request=InferenceRequest(
            request_id=f"request-{task_id}",
            model_name="demo",
            payload={},
            caller="test",
        ),
    )


def test_submit_stores_and_queues_record() -> None:
    backend = InMemoryTaskBackend()
    record = _record("task-1")

    backend.submit(record)

    assert backend.get("task-1") == record
    assert backend.pull() == "task-1"
    assert [record.task_id for record in backend.list_all()] == ["task-1"]


@pytest.mark.parametrize(
    "records, expected_order",
    [
        (
            [("low", 0), ("high-1", 10), ("high-2", 10)],
            ["high-1", "high-2", "low"],
        ),
    ],
)
def test_priority_and_fifo_ordering(
    records: list[tuple[str, int]],
    expected_order: list[str],
) -> None:
    backend = InMemoryTaskBackend()
    for task_id, priority in records:
        backend.submit(_record(task_id), priority=priority)

    assert [backend.pull() for _ in records] == expected_order


def test_duplicate_task_id_is_rejected_without_requeueing() -> None:
    backend = InMemoryTaskBackend()
    backend.submit(_record("task-1"))

    with pytest.raises(ValueError, match="task-1"):
        backend.submit(_record("task-1"))

    assert backend.pull() == "task-1"
    assert backend.pull() is None


def test_update_status_replaces_record() -> None:
    backend = InMemoryTaskBackend()
    backend.submit(_record("task-1"))

    updated = backend.update_status(
        "task-1",
        TaskStatus.FAILED,
        error_code=ErrorCode.INTERNAL_ERROR,
        error_message="failed",
    )

    assert updated.status == TaskStatus.FAILED
    assert updated.error_code == ErrorCode.INTERNAL_ERROR
    assert backend.get("task-1") == updated


def test_terminal_reads_return_isolated_copies() -> None:
    backend = InMemoryTaskBackend()
    backend.submit(_record("task-1"))
    backend.update_status(
        "task-1",
        TaskStatus.COMPLETED,
        result=InferenceResult(request_id="task-1", data={"answer": 42}),
    )

    fetched = backend.get("task-1", copy=True)
    assert fetched is not None
    fetched.request.payload["mutated"] = True
    fetched.result.data["answer"] = 0

    listed = backend.list_all()[0]
    listed.request.payload["listed"] = True

    first_terminal = backend.read("task-1")
    assert first_terminal is not None
    first_terminal.request.payload["terminal"] = True
    first_terminal.result.data["answer"] = 1

    second_terminal = backend.read("task-1")
    assert second_terminal is not None
    assert second_terminal.request.payload == {}
    assert second_terminal.result.data == {"answer": 42}


@pytest.mark.parametrize("status", [TaskStatus.COMPLETED, TaskStatus.FAILED])
def test_read_retains_terminal_record_by_default(status: TaskStatus) -> None:
    backend = InMemoryTaskBackend()
    backend.submit(_record("task-1"))
    backend.update_status(
        "task-1",
        status,
        result=InferenceResult(request_id="task-1", data={"answer": 42}),
    )
    updated_at = backend.get("task-1", copy=True).updated_at  # type: ignore[union-attr]

    first = backend.read("task-1")
    second = backend.read("task-1")

    assert first is not None
    assert first.status == status
    assert first.result is not None
    assert first.result.data == {"answer": 42}
    assert second == first
    assert backend.get("task-1") is not None
    assert (
        backend.get("task-1", copy=True).updated_at == updated_at
    )  # type: ignore[union-attr]


def test_read_can_atomically_consume_terminal_record() -> None:
    backend = InMemoryTaskBackend()
    backend.submit(_record("task-1"))
    backend.update_status("task-1", TaskStatus.COMPLETED)

    consumed = backend.read("task-1", consume=True)

    assert consumed is not None
    assert backend.read("task-1", consume=True) is None
    assert backend.get("task-1") is None


@pytest.mark.parametrize("status", [TaskStatus.PENDING, TaskStatus.RUNNING])
def test_read_ignores_non_terminal_record(status: TaskStatus) -> None:
    backend = InMemoryTaskBackend()
    backend.submit(_record("task-1"))
    if status == TaskStatus.RUNNING:
        backend.update_status("task-1", status)

    assert backend.read("task-1") is None
    assert backend.read("task-1", consume=True) is None
    assert backend.get("task-1") is not None


def test_concurrent_consumers_receive_terminal_record_once() -> None:
    backend = InMemoryTaskBackend()
    backend.submit(_record("task-1"))
    backend.update_status("task-1", TaskStatus.COMPLETED)
    barrier = threading.Barrier(10)
    consumed: list[TaskRecord | None] = []
    lock = threading.Lock()

    def consume() -> None:
        barrier.wait()
        record = backend.read("task-1", consume=True)
        with lock:
            consumed.append(record)

    threads = [threading.Thread(target=consume) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(record is not None for record in consumed) == 1


def test_default_retention_keeps_fifty_newest_terminal_records() -> None:
    backend = InMemoryTaskBackend()

    for index in range(51):
        task_id = f"task-{index:02d}"
        backend.submit(_record(task_id))
        backend.update_status(task_id, TaskStatus.COMPLETED)

    retained_ids = {record.task_id for record in backend.list_all()}
    assert len(retained_ids) == 50
    assert "task-00" not in retained_ids
    assert "task-50" in retained_ids


def test_retention_evicts_oldest_finish_time_not_oldest_submission() -> None:
    backend = InMemoryTaskBackend(max_retained_terminal_tasks=2)
    backend.submit(_record("submitted-first"))
    backend.submit(_record("finished-first"))
    backend.submit(_record("finished-last"))

    backend.update_status("finished-first", TaskStatus.COMPLETED)
    backend.update_status("submitted-first", TaskStatus.COMPLETED)
    backend.update_status("finished-last", TaskStatus.FAILED)

    retained_ids = {record.task_id for record in backend.list_all()}
    assert retained_ids == {"submitted-first", "finished-last"}


def test_retention_evicts_read_record_before_older_unread_record() -> None:
    backend = InMemoryTaskBackend(max_retained_terminal_tasks=2)
    for task_id in ("unread-old", "read-newer", "latest"):
        backend.submit(_record(task_id))

    backend.update_status("unread-old", TaskStatus.COMPLETED)
    backend.update_status("read-newer", TaskStatus.COMPLETED)
    assert backend.read("read-newer") is not None
    backend.update_status("latest", TaskStatus.COMPLETED)

    retained_ids = {record.task_id for record in backend.list_all()}
    assert retained_ids == {"unread-old", "latest"}


def test_retention_evicts_oldest_finished_record_within_read_records() -> None:
    backend = InMemoryTaskBackend(max_retained_terminal_tasks=2)
    for task_id in ("first", "second", "third"):
        backend.submit(_record(task_id))

    backend.update_status("first", TaskStatus.COMPLETED)
    backend.update_status("second", TaskStatus.COMPLETED)
    assert backend.read("first") is not None
    assert backend.read("second") is not None
    assert backend.read("first") is not None
    backend.update_status("third", TaskStatus.COMPLETED)

    retained_ids = {record.task_id for record in backend.list_all()}
    assert retained_ids == {"second", "third"}


def test_repeated_terminal_update_does_not_change_finish_order() -> None:
    backend = InMemoryTaskBackend(max_retained_terminal_tasks=2)
    for task_id in ("first", "second", "third"):
        backend.submit(_record(task_id))

    backend.update_status(
        "first",
        TaskStatus.COMPLETED,
        result=InferenceResult(request_id="first", data={"version": 1}),
    )
    backend.update_status("second", TaskStatus.COMPLETED)
    backend.update_status(
        "first",
        TaskStatus.COMPLETED,
        result=InferenceResult(request_id="first", data={"version": 2}),
    )
    backend.update_status("third", TaskStatus.COMPLETED)

    retained_ids = {record.task_id for record in backend.list_all()}
    assert retained_ids == {"second", "third"}


def test_zero_retention_limit_keeps_all_unconsumed_terminal_records() -> None:
    backend = InMemoryTaskBackend(max_retained_terminal_tasks=0)

    for index in range(3):
        task_id = f"task-{index}"
        backend.submit(_record(task_id))
        backend.update_status(task_id, TaskStatus.COMPLETED)

    assert len(backend.list_all()) == 3


def test_retention_limit_excludes_pending_and_running_records() -> None:
    backend = InMemoryTaskBackend(max_retained_terminal_tasks=1)
    backend.submit(_record("pending"))
    backend.submit(_record("running"))
    backend.update_status("running", TaskStatus.RUNNING)
    backend.submit(_record("old-terminal"))
    backend.update_status("old-terminal", TaskStatus.COMPLETED)
    backend.submit(_record("new-terminal"))
    backend.update_status("new-terminal", TaskStatus.FAILED)

    retained = {record.task_id: record.status for record in backend.list_all()}
    assert retained == {
        "pending": TaskStatus.PENDING,
        "running": TaskStatus.RUNNING,
        "new-terminal": TaskStatus.FAILED,
    }


def test_concurrent_submission_is_safe() -> None:
    backend = InMemoryTaskBackend()
    barrier = threading.Barrier(10)
    errors: list[Exception] = []

    def submit(batch: int) -> None:
        barrier.wait()
        for index in range(50):
            try:
                backend.submit(_record(f"{batch}-{index}"))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

    threads = [threading.Thread(target=submit, args=(batch,)) for batch in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(backend.list_all()) == 500
