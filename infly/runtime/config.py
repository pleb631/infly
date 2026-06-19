from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal


def _require_nonempty_text(value: str, *, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _require_int_at_least(value: int, minimum: int, *, field_name: str) -> int:
    if value < minimum:
        raise ValueError(f"{field_name} must be greater than or equal to {minimum}")
    return value


def _require_float_greater_than(
    value: float,
    minimum: float,
    *,
    field_name: str,
) -> float:
    if value <= minimum:
        raise ValueError(f"{field_name} must be greater than {minimum}")
    return value


def _require_float_at_least(
    value: float,
    minimum: float,
    *,
    field_name: str,
) -> float:
    if value < minimum:
        raise ValueError(f"{field_name} must be greater than or equal to {minimum}")
    return value


@dataclass(slots=True)
class WorkerSafetyPolicy:
    mode: Literal["shutdown", "degrade", "restart"] = "degrade"
    restart_limit: int = 3
    restart_window_seconds: float = 60
    restart_backoff_seconds: float = 1

    def __post_init__(self) -> None:
        if self.mode not in {"shutdown", "degrade", "restart"}:
            raise ValueError(
                "mode must be one of 'shutdown', 'degrade', or 'restart'"
            )
        self.restart_limit = _require_int_at_least(
            self.restart_limit,
            0,
            field_name="restart_limit",
        )
        self.restart_window_seconds = _require_float_greater_than(
            self.restart_window_seconds,
            0,
            field_name="restart_window_seconds",
        )
        self.restart_backoff_seconds = _require_float_at_least(
            self.restart_backoff_seconds,
            0,
            field_name="restart_backoff_seconds",
        )


@dataclass(slots=True)
class WorkerGroup:
    name: str
    device: str
    process_count: int = 1
    handlers: list[str] = field(default_factory=list)
    environment: Mapping[str, str] = field(default_factory=dict)
    safety: WorkerSafetyPolicy = field(default_factory=WorkerSafetyPolicy)

    def __post_init__(self) -> None:
        self.name = _require_nonempty_text(self.name, field_name="name")
        self.device = _require_nonempty_text(self.device, field_name="device")
        self.process_count = _require_int_at_least(
            self.process_count,
            1,
            field_name="process_count",
        )
        if any(not handler_name.strip() for handler_name in self.handlers):
            raise ValueError("handler names must not be empty")
        if len(self.handlers) != len(set(self.handlers)):
            raise ValueError("handlers must not contain duplicates")
        if "INFLY_DEVICE" in self.environment:
            raise ValueError("environment key 'INFLY_DEVICE' is reserved")
        self.environment = dict(self.environment)
        if isinstance(self.safety, Mapping):
            self.safety = WorkerSafetyPolicy(**self.safety)
        elif not isinstance(self.safety, WorkerSafetyPolicy):
            raise TypeError("safety must be a WorkerSafetyPolicy or mapping")


@dataclass(slots=True)
class StrategyConfig:
    default_sdk_strategy: str = "embedded_process_pool"
    worker_groups: list[WorkerGroup] = field(default_factory=list)
    embedded_pool_startup_timeout_seconds: float = 300

    def __post_init__(self) -> None:
        self.worker_groups = [
            group if isinstance(group, WorkerGroup) else WorkerGroup(**group)
            for group in self.worker_groups
        ]
        self.embedded_pool_startup_timeout_seconds = _require_float_greater_than(
            self.embedded_pool_startup_timeout_seconds,
            0,
            field_name="embedded_pool_startup_timeout_seconds",
        )


@dataclass(slots=True)
class SchedulerConfig:
    max_outstanding_tasks: int = 50
    num_workers: int = 2
    max_retained_terminal_tasks: int = 50

    def __post_init__(self) -> None:
        self.max_outstanding_tasks = _require_int_at_least(
            self.max_outstanding_tasks,
            1,
            field_name="max_outstanding_tasks",
        )
        self.num_workers = _require_int_at_least(
            self.num_workers,
            1,
            field_name="num_workers",
        )
        self.max_retained_terminal_tasks = _require_int_at_least(
            self.max_retained_terminal_tasks,
            0,
            field_name="max_retained_terminal_tasks",
        )



__all__ = [
    "SchedulerConfig",
    "StrategyConfig",
    "WorkerGroup",
    "WorkerSafetyPolicy",
]
