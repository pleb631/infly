from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future
from typing import Any, Protocol

from infly.core.contracts import (
    TaskRecord,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from infly.core.errors import ErrorCode


class HandlerProtocol(Protocol):
    def handle(self, input: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


class HandlerFactory(Protocol):
    def __call__(
        self,
        init_context: Mapping[str, Any],
        **kwargs: Any,
    ) -> HandlerProtocol:
        ...


class ExecutionStrategy(Protocol):
    def execute(self, request: TaskRequest) -> Future[TaskResult]:
        ...

    def close(self) -> None:
        ...


class TaskBackend(Protocol):
    def submit(self, record: TaskRecord, priority: int = 0) -> None:
        ...

    def pull(self) -> str | None:
        ...

    def get(self, task_id: str, copy: bool = False) -> TaskRecord | None:
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
        result: TaskResult | None = None,
        error_code: ErrorCode | None = None,
        error_message: str | None = None,
    ) -> TaskRecord:
        ...

    def list_all(self) -> list[TaskRecord]:
        ...


__all__ = [
    "ExecutionStrategy",
    "HandlerFactory",
    "HandlerProtocol",
    "TaskBackend",
]
