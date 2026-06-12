from __future__ import annotations

import datetime
from enum import StrEnum
from typing import Any, Self
from pydantic import BaseModel, Field

from infly.core.errors import ErrorCode


class InferenceRequest(BaseModel):
    request_id: str
    model_name: str
    payload: dict[str, Any]
    caller: str
    metadata: dict[str, Any] = Field(default_factory=dict)



class InferenceResult(BaseModel):
    request_id: str
    data: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)

class TaskStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class TaskRecord(BaseModel):
    task_id: str
    request: InferenceRequest
    status: TaskStatus = TaskStatus.PENDING
    result: dict[str, Any] | None = None
    error_code: ErrorCode | None = None
    error_message: str | None = None
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )
    updated_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )


class TaskQueryResponse(BaseModel):
    task_id: str
    status: TaskStatus
    result: dict[str, Any] | None = None
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
