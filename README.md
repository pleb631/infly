# infly

`infly` 是一个面向 Python 3.12+ 的轻量推理运行时，适合把模型调用、规则处理或其他可执行逻辑封装成统一的 `handler`，再交给调度器提交、执行、查询和观测。

它提供的不是一整套 HTTP 服务，而是一层清晰的运行时内核：你可以把它接到 API、任务系统、批处理流程或内部平台里，用一致的方式管理任务生命周期、进程池执行和运行时观测。

## 核心能力

- `HandlerDefinition` / `HandlerRegistry`
  用来声明、注册和管理可执行 handler。
- `TaskRequest` / `TaskResult` / `TaskRecord` / `TaskStatus`
  用来描述任务请求、执行结果和任务状态。
- `TaskScheduler`
  负责任务提交、排队、等待和查询。
- `ProcessPoolStrategy`
  负责在独立 worker 进程中执行 handler。
- `WorkerGroup` / `WorkerSafetyPolicy` / `SchedulerConfig`
  用来控制 worker 分组、并发度、容量限制和故障策略。
- `RuntimeInstrumentation`
  提供 health snapshot、metrics snapshot、Prometheus 文本和 trace event。
- `PlatformError` / `ErrorCode`
  提供统一错误模型。

## 安装

安装项目本体：

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

先定义一个 handler factory。`entrypoint` 需要使用 `module:SymbolName` 形式，指向一个可调用对象，并返回一个带 `handle(input)` 方法的实例。

```python
from collections.abc import Mapping
from typing import Any


class EchoHandler:
    def __init__(self, init_context: Mapping[str, Any], **init_kwargs: Any) -> None:
        self.init_context = dict(init_context)
        self.init_kwargs = dict(init_kwargs)

    def handle(self, input: Mapping[str, Any]) -> dict[str, Any]:
        prefix = str(self.init_context.get("prefix", ""))
        return {"echo": f"{prefix}{input['text']}"}


def build_echo_handler(
    init_context: Mapping[str, Any],
    **init_kwargs: Any,
) -> EchoHandler:
    return EchoHandler(init_context, **init_kwargs)
```

然后注册 handler，并通过顶层公共 API 启动运行时：

```python
from infly import (
    HandlerDefinition,
    HandlerRegistry,
    ProcessPoolStrategy,
    SchedulerConfig,
    TaskRequest,
    TaskScheduler,
    WorkerGroup,
)


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
        num_threads=2,
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

`submit()` 会返回一个 `task_id`，之后你可以按需查询当前状态或等待终态结果：

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

常见查询方式：

- `query(task_id)`：立即返回当前快照
- `query(task_id, wait=True)`：等待任务进入终态
- `query(task_id, wait=True, consume=True)`：读取一次后从后端消费掉终态记录

## 可观测性

`infly` 内置的是运行时观测能力，而不是预制的 `/health` 或 `/metrics` HTTP 服务。你可以从 Python API 直接拿到 health、metrics 和 trace event，再按自己的接入层暴露出来。

```python
from infly import (
    HandlerDefinition,
    HandlerRegistry,
    ProcessPoolStrategy,
    RuntimeInstrumentation,
    SchedulerConfig,
    TaskRequest,
    TaskScheduler,
    WorkerGroup,
)


registry = HandlerRegistry()
registry.add(
    HandlerDefinition(
        handler_name="echo",
        entrypoint="my_handlers:build_echo_handler",
    )
)

instrumentation = RuntimeInstrumentation()
instrumentation.add_trace_sink(
    lambda event: print(event.name, event.task_key, event.trace_id)
)

scheduler = TaskScheduler(
    ProcessPoolStrategy(
        registry,
        [WorkerGroup(name="cpu", device="cpu", process_count=2, handlers=["echo"])],
    ),
    scheduler_config=SchedulerConfig(num_threads=2),
    instrumentation=instrumentation,
)

scheduler.start()
try:
    result = scheduler.submit_and_wait(
        TaskRequest(
            task_key="health-demo",
            handler_name="echo",
            input={"text": "hello"},
            caller="api",
            metadata={"trace_id": "trace-health-demo"},
        )
    )
    print(result.output)

    health = scheduler.health_snapshot()
    print(health.status)
    print(health.backend_status_counts)

    metrics = instrumentation.metrics_snapshot()
    print(metrics.submitted_total, metrics.completed_total)

    prometheus_text = instrumentation.render_prometheus_text()
    print(prometheus_text)
finally:
    scheduler.stop()
```

当前 trace sink 会收到这些生命周期事件：

- `task.submitted`
- `task.started`
- `task.completed`
- `task.failed`

如果你要对接自己的服务层，通常可以这样映射：

- `/health`：返回 `scheduler.health_snapshot()`
- `/metrics`：返回 `instrumentation.render_prometheus_text()`
- tracing：在 `add_trace_sink()` 里把 `TraceEvent` 转成自己的 span 或结构化事件

## 错误处理

常见 `ErrorCode` 包括：

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

调用失败通常会抛出 `PlatformError`，可以通过 `exc.code` 读取错误码。

## Demo 脚本

仓库里带了两个 demo，适合快速理解调度和观测链路：

```bash
python scripts/demo_observability.py
python scripts/demo_quickstart.py
```

`demo_observability.py` 会展示：

- 一个成功任务的执行结果
- 一个失败任务的错误信息
- scheduler / strategy health snapshot
- metrics snapshot 和 Prometheus 文本
- `task.submitted`、`task.started`、`task.completed`、`task.failed` 这些 trace event

`demo_quickstart.py` 会展示：

- 注册两个 handler
- 用 `ProcessPoolStrategy` 和 `WorkerGroup` 启动 worker
- 同步提交一个成功任务
- 异步提交一个任务并用 `query(..., wait=True)` 读取结果
- 提交一个失败任务并观察错误传播
- 打印 health / metrics / Prometheus / trace 输出

这些 demo 只是示例支撑代码，不是顶层公共 API 的一部分。
