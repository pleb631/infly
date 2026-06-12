import datetime


from infly.core.contracts import (
    InferenceRequest,
    TaskQueryResponse,
    TaskRecord,
    TaskStatus,
)
from infly.core.errors import ErrorCode, PlatformError


def _request() -> InferenceRequest:
    return InferenceRequest(
        request_id="req-1",
        model_name="demo",
        payload={"text": "hello"},
        caller="test",
    )


def test_platform_error_exposes_code_and_message() -> None:
    error = PlatformError(ErrorCode.NOT_FOUND, "not found")
    assert error.code == ErrorCode.NOT_FOUND
    assert error.message == "not found"
    assert str(error) == "not found"


def test_task_record_defaults() -> None:
    record = TaskRecord(task_id="task-1", request=_request())
    assert record.status == TaskStatus.PENDING
    assert record.result is None
    assert record.error_code is None
    assert isinstance(record.created_at, datetime.datetime)


def test_task_query_response_from_record() -> None:
    record = TaskRecord(
        task_id="task-1",
        request=_request(),
        status=TaskStatus.COMPLETED,
        result={"answer": 42},
    )

    response = TaskQueryResponse.from_record(record)

    assert response.task_id == "task-1"
    assert response.status == TaskStatus.COMPLETED
    assert response.result == {"answer": 42}
    assert response.error_code is None
    assert response.error_message is None
