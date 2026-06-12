from typing import Literal

from pydantic import BaseModel, Field, field_validator


class WorkerSafetyPolicy(BaseModel):
    mode: Literal["shutdown", "degrade", "restart"] = "degrade"
    restart_limit: int = Field(default=3, ge=0)
    restart_window_seconds: float = Field(default=60, gt=0)
    restart_backoff_seconds: float = Field(default=1, ge=0)


class WorkerGroup(BaseModel):
    name: str
    device: str
    process_count: int = Field(default=1, ge=1)
    models: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    safety: WorkerSafetyPolicy = Field(default_factory=WorkerSafetyPolicy)

    @field_validator("name", "device")
    @classmethod
    def require_nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value

    @field_validator("models")
    @classmethod
    def validate_models(cls, value: list[str]) -> list[str]:
        if any(not model_name.strip() for model_name in value):
            raise ValueError("model names must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("models must not contain duplicates")
        return value

    @field_validator("environment")
    @classmethod
    def reject_reserved_environment(cls, value: dict[str, str]) -> dict[str, str]:
        if "INFLY_DEVICE" in value:
            raise ValueError("environment key 'INFLY_DEVICE' is reserved")
        return value


class StrategyConfig(BaseModel):
    default_sdk_strategy: str = "embedded_process_pool"
    worker_groups: list[WorkerGroup] = Field(default_factory=list)
    embedded_pool_startup_timeout_seconds: float = Field(default=300, gt=0)


class SchedulerConfig(BaseModel):
    max_outstanding_tasks: int = Field(default=50, ge=1)
    num_workers: int = Field(default=2, ge=1)
    max_retained_terminal_tasks: int = Field(default=50, ge=0)


__all__ = [
    "SchedulerConfig",
    "StrategyConfig",
    "WorkerGroup",
    "WorkerSafetyPolicy",
]
