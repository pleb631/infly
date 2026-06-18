from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Self
from infly.core.errors import ErrorCode



@dataclass(slots=True, frozen=True)
class InferenceRequest:
    request_id: str
    model_name: str
    payload: Mapping[str, Any]
    caller: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(slots=True, frozen=True)
class InferenceResult:
    request_id: str
    data: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

class TaskStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(slots=True)
class TaskRecord:
    task_id: str
    request: InferenceRequest
    status: TaskStatus = TaskStatus.PENDING
    result: InferenceResult | None = None
    error_code: ErrorCode | None = None
    error_message: str | None = None
    created_at: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )
    updated_at: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )


@dataclass(slots=True, frozen=True)
class TaskQueryResponse:
    task_id: str
    status: TaskStatus
    result: InferenceResult | None = None
    error_code: ErrorCode | None = None
    error_message: str | None = None

    @classmethod
    def from_record(cls, record: TaskRecord) -> Self:
        return cls(
            task_id=record.task_id,
            status=record.status,
            result=record.result,
            error_code=record.error_code,
            error_message=record.error_message,
        )


__all__ = [
    "InferenceRequest",
    "InferenceResult",
    "TaskQueryResponse",
    "TaskRecord",
    "TaskStatus",
]
