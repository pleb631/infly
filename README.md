# infly

`infly` 是一个面向 Python 应用的嵌入式推理运行时。它把模型注册、模型实例缓存、任务调度、状态查询、多进程执行和日志路由封装在一套 Python API 里，适合直接集成到服务进程中。

当前仓库没有独立 CLI，主要通过代码调用。

## 适用场景

- 你已经有 Python 模型类，想挂到统一的请求接口上
- 你需要带优先级、并发上限和任务状态查询的调度层
- 你希望模型跑在独立进程里，避免阻塞主进程
- 你希望按 worker、任务类别和 logger 名称落盘日志

## 环境要求

- Python 3.12+

## 安装

```bash
pip install -e .
```

安装开发依赖：

```bash
pip install -e ".[dev]"
```

使用 `uv`：

```bash
uv sync
```

## 核心对象

- `ModelDefinition`
  定义模型名、指向类或工厂函数的 `class_path`、构造参数和元数据
- `ModelRegistry`
  负责模型定义的注册、替换和查找
- `InferenceRequest`
  推理请求，字段为 `request_id`、`model_name`、`payload`、`caller`、`metadata`
- `InferenceResult`
  推理结果，字段为 `request_id`、`data`、`diagnostics`
- `EmbeddedProcessPoolStrategy`
  多进程执行策略，负责 worker 生命周期和请求分发
- `TaskScheduler`
  调度器，负责任务入队、并发控制、等待和状态查询
- `PlatformError`
  统一异常类型，附带 `ErrorCode`

## 数据约定

当前版本以 `Mapping` 作为请求和结果字段的约定：

- 模型接收 `payload: Mapping[str, Any]`
- 模型返回 `Mapping[str, Any]`
- `InferenceRequest.payload` / `metadata`
- `InferenceResult.data` / `diagnostics`

这几个对象不会像旧版 Pydantic 模型那样自动深拷贝。对调用方的约定是：

- `submit()` 之后，把传入的 `payload` / `metadata` 视为只读
- 模型返回的 `data` / `diagnostics` 也应视为运行时拥有的数据

`TaskScheduler.query()` 和 `TaskBackend.read()` 返回给调用方的终态记录会做隔离复制，避免调用方修改回写到后端存储。

## 快速开始

最常见的接入流程：

1. 定义模型类
2. 注册模型
3. 创建执行策略
4. 创建调度器并提交任务

### 1. 定义模型

运行时最终需要一个模型实例，并满足以下约定：

- 模型实例需要实现 `predict(payload) -> Mapping[str, Any]`
- `class_path` 指向的模块符号可以是类或工厂函数，并且应接收
  `module_dict` 与 `**kwargs` 后返回该模型实例

```python
from collections.abc import Mapping
from typing import Any


class EchoModel:
    def __init__(self, module_dict: Mapping[str, Any], prefix: str = "") -> None:
        self.prefix = prefix
        self.module_dict = module_dict

    def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "echo": f"{self.prefix}{payload['text']}",
            "gpu": self.module_dict.get("gpu"),
        }
```

### 2. 注册模型

`class_path` 使用 `module:SymbolName` 格式，可指向类或工厂函数。

```python
from infly.core.models import ModelDefinition
from infly.runtime.registry import ModelRegistry


registry = ModelRegistry()
registry.add(
    ModelDefinition(
        model_name="echo",
        class_path="my_models:EchoModel",
        module_dict={"gpu": 0},
        kwargs={"prefix": "[demo] "},
        metadata={"team": "search"},
    )
)
```

保留现有类写法的同时，也支持显式工厂函数：

```python
from collections.abc import Mapping
from typing import Any


def build_echo_model(
    module_dict: Mapping[str, Any],
    **kwargs: Any,
) -> EchoModel:
    return EchoModel(module_dict, **kwargs)


registry.add(
    ModelDefinition(
        model_name="echo-factory",
        class_path="my_models:build_echo_model",
        module_dict={"gpu": 1},
        kwargs={"prefix": "[factory] "},
    )
)
```

### 3. 创建 worker 组和执行策略

`EmbeddedProcessPoolStrategy` 会把请求发送到独立进程。每个 `WorkerGroup` 可以绑定设备、进程数、环境变量和故障策略。

```python
from infly.runtime.config import WorkerGroup
from infly.runtime.strategy.embedded_process_pool import EmbeddedProcessPoolStrategy


strategy = EmbeddedProcessPoolStrategy(
    registry,
    [
        WorkerGroup(
            name="cpu",
            device="cpu",
            process_count=2,
            models=["echo"],
            environment={"OMP_NUM_THREADS": "1"},
            safety={"mode": "restart", "restart_limit": 3},
        )
    ],
)
```

说明：

- `safety` 可以直接传 `WorkerSafetyPolicy`，也可以传 `dict`
- worker 进程里会自动注入 `INFLY_DEVICE`
- runtime 会把 `worker_context` 注入到模型的 `module_dict` 顶层映射里

### 4. 创建调度器并提交任务

```python
from infly.core.contracts import InferenceRequest
from infly.runtime.config import SchedulerConfig
from infly.runtime.scheduler import TaskScheduler


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
        InferenceRequest(
            request_id="req-1",
            model_name="echo",
            payload={"text": "hello"},
            caller="api",
            metadata={"trace_id": "trace-1"},
        ),
        timeout_seconds=30,
    )
    print(result.data)
    print(result.diagnostics)
finally:
    scheduler.stop()
```

## 查询任务状态

如果你不想阻塞等待，可以先 `submit()`，再单独查询：

```python
task_id = scheduler.submit(
    InferenceRequest(
        request_id="req-2",
        model_name="echo",
        payload={"text": "async"},
        caller="api",
    ),
    priority=10,
)

response = scheduler.query(task_id)
print(response.status)
```

阻塞等待终态：

```python
response = scheduler.query(task_id, wait=True, timeout_seconds=10)
print(response.status)
print(response.result.data if response.result is not None else None)
```

读取并消费终态记录：

```python
response = scheduler.query(task_id, wait=True, consume=True)
```

`consume=True` 适合“结果只取一次”的场景。记录被消费后，再查同一个 `task_id` 会得到 `NOT_FOUND`。

## 调度与保留策略

`SchedulerConfig`：

- `max_outstanding_tasks`
  允许同时处于 `PENDING` 或 `RUNNING` 的最大任务数
- `num_workers`
  调度线程数，不是模型进程数
- `max_retained_terminal_tasks`
  后端保留的终态任务数，`0` 表示不自动淘汰

调度器默认使用 `InMemoryTaskBackend`。

后端行为：

- 支持优先级队列
- 支持终态记录保留和淘汰
- `read()` 默认返回终态快照
- `read(..., consume=True)` 会原子消费

## worker 配置

`WorkerGroup` 字段：

- `name`
  组名，不能为空，必须唯一
- `device`
  设备标识，例如 `cpu`、`cuda:0`
- `process_count`
  该组启动多少个 worker 进程，最小值为 `1`
- `models`
  该组可服务的模型名列表；留空表示服务注册表中的全部模型
- `environment`
  注入到 worker 进程的环境变量；`INFLY_DEVICE` 是保留键
- `safety`
  故障策略

`WorkerSafetyPolicy.mode` 支持：

- `degrade`
  默认模式，worker 挂掉后不重启，容量下降
- `restart`
  worker 挂掉后按策略重启
- `shutdown`
  任意 worker 异常退出后关闭整个池

重启相关字段：

- `restart_limit`
- `restart_window_seconds`
- `restart_backoff_seconds`

## 模型加载与缓存

`InferenceService` 会按 `ModelDefinition.cache_key` 缓存模型实例。缓存键会综合以下字段：

- `model_name`
- `class_path`
- `module_dict`
- `kwargs`
- `metadata`

当你用同名模型重新注册不同定义时：

- 注册表会替换旧定义
- 下次推理会按新的 `cache_key` 重新加载模型
- 旧实例会从活动缓存里移除

## 日志

推荐在程序启动时先配置日志，再创建策略和调度器。

### 直接配置

```python
from infly.runtime.log import configure_logging


configure_logging(
    log_root="logs/infly",
    log_level="INFO",
    save_days=30,
    log_format="%(asctime)s - %(name)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s",
)
```

默认值：

- `log_root=logs/infly`
- `log_level=INFO`
- `save_days=30`
- `log_format=%(asctime)s - %(name)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s`

### 直接构造 `LoggingSettings`

如果你需要显式传给 `setup_main_logging()` 或其他封装，也可以直接构造：

```python
from infly.runtime.log import LoggingSettings


settings = LoggingSettings(
    log_root="logs/custom",
    log_level="WARNING",
    save_days=7,
)
```

`log_root` 支持 `str` 或 `Path`，`log_level` 支持字符串级别名或整数。

### 主进程注册 sink

```python
def on_log(record):
    print(record.levelname, record.getMessage())


pool = EmbeddedProcessPoolStrategy(registry, [WorkerGroup(name="cpu", device="cpu")])
pool.log_manager.add_sink(on_log)
```

### 手动接入子进程日志

如果你自己写多进程入口，可以显式安装队列日志：

```python
from infly.runtime.log import get_logger, setup_worker_logging


def worker_main(log_queue) -> None:
    setup_worker_logging(log_queue)
    log = get_logger("worker", category="worker")
    log.info("worker started")
```

如果你使用 `EmbeddedProcessPoolStrategy`，worker 侧日志初始化已经由 runtime 自动完成。

## 错误处理

运行时统一抛 `PlatformError`，常见错误码：

- `MODEL_NOT_FOUND`
- `NOT_FOUND`
- `OVERLOADED`
- `TIMEOUT`
- `WORKER_UNAVAILABLE`
- `INVALID_ARGUMENT`
- `INVALID_CONFIGURATION`
- `INVALID_STATE`
- `INVALID_REQUEST`
- `INTERNAL_ERROR`

示例：

```python
from infly.core.errors import PlatformError


try:
    scheduler.submit_and_wait(request, timeout_seconds=1)
except PlatformError as exc:
    print(exc.code, exc.message)
```

## 测试

```bash
pytest -q
```

或：

```bash
uv run pytest -q
```

## 目录结构

- `infly/core`
  请求/结果契约、错误码、协议、模型定义
- `infly/runtime`
  调度器、任务后端、模型注册、模型加载、日志、执行策略
- `tests`
  单元测试和集成测试

## 当前边界

- 当前仓库不提供独立 CLI
- 默认任务后端是内存实现，不是持久化队列
- 当前内置执行策略主要是 `EmbeddedProcessPoolStrategy`
