from .core.contracts import TaskRecord, TaskRequest, TaskResult, TaskStatus
from .core.errors import ErrorCode, PlatformError
from .core.handlers import HandlerDefinition
from .core.ports import ExecutionStrategy, HandlerFactory, HandlerProtocol, TaskBackend
from .runtime.config import (
    SchedulerConfig,
    WorkerGroup,
    WorkerSafetyPolicy,
)
from .runtime.executor import HandlerExecutor
from .runtime.log import configure_logging, get_logger
from .runtime.observability import (
    HealthStatus,
    RuntimeInstrumentation,
    RuntimeMetricsSnapshot,
    SchedulerHealthSnapshot,
    StrategyHealthSnapshot,
    TraceEvent,
)
from .runtime.registry import HandlerRegistry
from .runtime.scheduler import TaskScheduler
from .runtime.strategy import ProcessPoolStrategy
from .runtime.task_backend import InMemoryTaskBackend

__all__ = [
    "ErrorCode",
    "ExecutionStrategy",
    "HandlerDefinition",
    "HandlerExecutor",
    "HandlerFactory",
    "HandlerProtocol",
    "HandlerRegistry",
    "HealthStatus",
    "InMemoryTaskBackend",
    "PlatformError",
    "ProcessPoolStrategy",
    "RuntimeInstrumentation",
    "RuntimeMetricsSnapshot",
    "SchedulerConfig",
    "SchedulerHealthSnapshot",
    "StrategyHealthSnapshot",
    "TaskBackend",
    "TaskRecord",
    "TaskRequest",
    "TaskResult",
    "TaskScheduler",
    "TaskStatus",
    "TraceEvent",
    "WorkerGroup",
    "WorkerSafetyPolicy",
    "configure_logging",
    "get_logger",
]


__version__ = "0.4.1"
