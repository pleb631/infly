from __future__ import annotations

from concurrent.futures import Future
from typing import Any, Protocol

from infly.core.contracts import (
    InferenceRequest,
    InferenceResult,
    TaskRecord,
    TaskStatus,
)
from infly.core.errors import ErrorCode


class ModelProtocol(Protocol):
    def __init__(self, module_dict: dict[str, Any], **kwargs: Any) -> None:
        ...

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


class ExecutionStrategy(Protocol):
    def execute(self, request: InferenceRequest) -> Future[InferenceResult]:
        ...

    def close(self) -> None:
        ...

class TaskBackend(Protocol):
    def submit(self, record: TaskRecord, priority: int = 0) -> None:
        ...

    def pull(self) -> str | None:
        ...

    def get(self, task_id: str) -> TaskRecord | None:
        ...

    def read(
        self,
        task_id: str,
        *,
        consume: bool = False,
    ) -> TaskRecord | None:
        ...

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, Any] | None = None,
        error_code: ErrorCode | None = None,
        error_message: str | None = None,
    ) -> TaskRecord:
        ...

    def list_all(self) -> list[TaskRecord]:
        ...


__all__ = [
    "ExecutionStrategy",
    "ModelProtocol",
    "TaskBackend",
]
