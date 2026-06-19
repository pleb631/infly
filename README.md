# infly

`infly` 是一个面向 Python 3.12+ 的轻量推理运行时。它提供 handler 注册、动态加载、任务调度、进程池执行、任务状态查询和结构化日志路由，适合把推理逻辑封装成可调度的运行单元。

## 主要能力

- `HandlerDefinition` / `HandlerRegistry`：注册和管理可执行 handler
- `TaskRequest` / `TaskResult` / `TaskRecord` / `TaskStatus`：描述任务请求、执行结果和生命周期
- `TaskScheduler`：负责提交、排队、查询和等待任务
- `ProcessPoolStrategy`：在独立进程中执行 handler
- `WorkerGroup` / `WorkerSafetyPolicy` / `SchedulerConfig`：控制 worker 分组、扩缩容和调度限制
- `ErrorCode` / `PlatformError`：统一错误表达

## 安装

开发环境安装：

```bash
pip install -e .
```

安装开发依赖：

```bash
pip install -e ".[dev]"
```

如果你使用 `uv`：

```bash
uv sync
```

## 快速开始

先定义一个 handler factory。`entrypoint` 需要指向 `module:SymbolName`，这个符号应当是一个可调用对象，并返回一个带 `handle(input)` 方法的实例。

```python
from collections.abc import Mapping
from typing import Any


class EchoHandler:
    def __init__(self, init_context: Mapping[str, Any], **init_kwargs: Any) -> None:
        self.init_context = init_context
        self.init_kwargs = init_kwargs

    def handle(self, input: Mapping[str, Any]) -> dict[str, Any]:
        prefix = str(self.init_context.get("prefix", ""))
        return {"echo": f"{prefix}{input['text']}"}


def build_echo_handler(
    init_context: Mapping[str, Any],
    **init_kwargs: Any,
) -> EchoHandler:
    return EchoHandler(init_context, **init_kwargs)
```

注册 handler，然后用进程池策略和调度器执行它：

```python
from infly.core.contracts import TaskRequest
from infly.core.handlers import HandlerDefinition
from infly.runtime.config import SchedulerConfig, WorkerGroup
from infly.runtime.registry import HandlerRegistry
from infly.runtime.scheduler import TaskScheduler
from infly.runtime.strategy import ProcessPoolStrategy


registry = HandlerRegistry()
registry.add(
    HandlerDefinition(
        handler_name="echo",
        entrypoint="my_handlers:build_echo_handler",
        init_context={"prefix": "[demo] "},
        metadata={"team": "search"},
    )
)

strategy = ProcessPoolStrategy(
    registry,
    [
        WorkerGroup(
            name="cpu",
            device="cpu",
            process_count=2,
            handlers=["echo"],
            environment={"OMP_NUM_THREADS": "1"},
        )
    ],
)

scheduler = TaskScheduler(
    strategy,
    scheduler_config=SchedulerConfig(
        max_outstanding_tasks=32,
        num_workers=2,
        max_retained_terminal_tasks=100,
    ),
)

scheduler.start()
try:
    result = scheduler.submit_and_wait(
        TaskRequest(
            task_key="req-1",
            handler_name="echo",
            input={"text": "hello"},
            caller="api",
            metadata={"trace_id": "trace-1"},
        ),
        timeout_seconds=30,
    )
    print(result.output)
    print(result.diagnostics)
finally:
    scheduler.stop()
```

## 查询任务

`submit()` 会返回一个 `task_id`，之后可以单独查询。

```python
task_id = scheduler.submit(
    TaskRequest(
        task_key="req-2",
        handler_name="echo",
        input={"text": "async"},
        caller="api",
    ),
    priority=10,
)

response = scheduler.query(task_id)
print(response.status)
```

常用的查询方式：

- `query(task_id)`：立即返回当前快照
- `query(task_id, wait=True)`：等待直到任务进入终态
- `query(task_id, wait=True, consume=True)`：读取一次后从后端移除

## 核心概念

### HandlerDefinition

`HandlerDefinition` 描述一个可加载的 handler。

- `handler_name`：调度时使用的名称
- `entrypoint`：`module:SymbolName` 形式的入口
- `init_context`：构造 handler 时传入的上下文
- `init_kwargs`：构造 handler 时传入的额外关键字参数
- `metadata`：附加元数据

`runtime_context` 是保留键，不应手动放进 `init_context`。进程池在 worker 侧会自动注入它。

### WorkerGroup

`WorkerGroup` 决定 worker 如何启动和分组。

- `name`：唯一的 worker 组名
- `device`：暴露给 worker 的设备标识，会写入 `INFLY_DEVICE`
- `process_count`：该组启动的 worker 进程数
- `handlers`：该组允许执行的 handler 名称列表。为空时表示执行所有已注册 handler
- `environment`：注入 worker 进程的额外环境变量
- `safety`：worker 退出后的重启或降级策略

### TaskScheduler

`TaskScheduler` 负责管理任务生命周期。

- `start()`：启动后台 worker 线程
- `submit()`：提交任务并返回 `task_id`
- `submit_and_wait()`：提交任务并等待最终结果
- `query()`：查询任务状态或结果
- `stop()`：停止调度器并关闭底层 strategy

### ProcessPoolStrategy

`ProcessPoolStrategy` 是内置的进程池执行策略。

- 会在独立进程中预加载 handler
- 会把 `group_name`、`worker_id` 和 `device` 注入 worker runtime context
- 会设置 worker 进程名，并启用结构化日志路由
- 支持 worker 崩溃后的重启或降级，具体取决于 `WorkerSafetyPolicy`

## 错误处理

常见的 `ErrorCode` 包括：

- `HANDLER_NOT_FOUND`
- `NOT_FOUND`
- `OVERLOADED`
- `TIMEOUT`
- `WORKER_UNAVAILABLE`
- `INVALID_ARGUMENT`
- `INVALID_CONFIGURATION`
- `INVALID_STATE`
- `INVALID_REQUEST`
- `INTERNAL_ERROR`

调用失败时通常会抛出 `PlatformError`，可通过 `exc.code` 读取错误码。

## 日志

运行时使用结构化日志，worker 侧会自动接入日志路由。相关实现位于 [infly/runtime/log.py](./infly/runtime/log.py)。

## 测试

```bash
pytest -q
```

## 项目结构

- `infly/core`：任务契约、handler 定义、协议类型和错误码
- `infly/runtime`：注册表、执行器、调度器、任务后端、进程池策略和日志
- `tests`：单元测试和集成测试
