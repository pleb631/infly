from .core.handlers import HandlerDefinition
from .core.ports import ExecutionStrategy, HandlerFactory, HandlerProtocol, TaskBackend
from .core.contracts import TaskRequest, TaskResult, TaskRecord, TaskStatus
from .core.errors import ErrorCode, PlatformError
from .runtime.strategy import ProcessPoolStrategy
from .runtime.config import (
    SchedulerConfig,
    StrategyConfig,
    WorkerGroup,
    WorkerSafetyPolicy,
)
from .runtime.log import get_logger, configure_logging
from .runtime.registry import HandlerRegistry
from .runtime.scheduler import TaskScheduler
from .runtime.executor import HandlerExecutor
from .runtime.task_backend import InMemoryTaskBackend


__all__ = [
    "HandlerDefinition",
    "ExecutionStrategy",
    "HandlerFactory",
    "HandlerProtocol",
    "TaskBackend",
    "TaskRequest",
    "TaskResult",
    "TaskRecord",
    "TaskStatus",
    "ErrorCode",
    "PlatformError",
    "ProcessPoolStrategy",
    "SchedulerConfig",
    "StrategyConfig",
    "WorkerGroup",
    "WorkerSafetyPolicy",
    "get_logger",
    "configure_logging",
    "HandlerRegistry",
    "TaskScheduler",
    "HandlerExecutor",
    "InMemoryTaskBackend",
]


__version__ = "0.3.1"
