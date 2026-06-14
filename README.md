# infly

`infly` 是一个面向 Python 应用的推理运行时。它把模型注册、任务调度、并发执行、失败回收和多进程模型推理封装起来，适合直接嵌入到你的服务里使用。

当前仓库没有独立 CLI，主要通过 Python API 集成。

## 适合什么场景

- 你已经有一个 Python 模型类，希望把它接到统一的请求/任务接口上
- 你需要一个带优先级、并发上限、状态查询的任务调度层
- 你希望模型运行在独立进程中，避免主进程被阻塞

## 环境要求

- Python 3.12+

## 安装

开发环境可以直接安装本项目：

```bash
pip install -e .
```

如果你还想安装测试依赖：

```bash
pip install -e ".[dev]"
```

也可以使用 `uv`：

```bash
uv sync
```

## 快速开始

最小接入流程有四步：

1. 定义一个模型类
2. 注册模型
3. 创建执行策略
4. 通过调度器提交任务并等待结果

### 1. 定义模型

模型类需要实现两个约定：

- `__init__(module_dict, **kwargs)`
- `predict(payload) -> dict`

```python
from typing import Any


class EchoModel:
    def __init__(self, module_dict: dict[str, Any], prefix: str = "") -> None:
        self.prefix = prefix

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "echo": f"{self.prefix}{payload['text']}",
        }
```

### 2. 注册模型

`class_path` 使用 `module:ClassName` 格式。

```python
from infly.core.models import ModelDefinition
from infly.runtime.registry import ModelRegistry


registry = ModelRegistry()
registry.add(
    ModelDefinition(
        model_name="echo",
        class_path="my_models:EchoModel",
        kwargs={"prefix": "[demo] "},
    )
)
```

### 3. 创建执行策略

如果你想让模型在独立进程中运行，可以使用 `EmbeddedProcessPoolStrategy`。

```python
from infly.runtime.config import WorkerGroup
from infly.runtime.strategy.embedded_process_pool import EmbeddedProcessPoolStrategy


strategy = EmbeddedProcessPoolStrategy(
    registry,
    [
        WorkerGroup(
            name="cpu",
            device="cpu",
            process_count=1,
            models=["echo"],
        )
    ],
)
```

### 4. 提交任务

```python
from infly.core.contracts import InferenceRequest
from infly.runtime.scheduler import TaskScheduler
from infly.runtime.config import SchedulerConfig


scheduler = TaskScheduler(
    strategy,
    scheduler_config=SchedulerConfig(
        max_outstanding_tasks=10,
        num_workers=1,
    ),
)

scheduler.start()
try:
    result = scheduler.submit_and_wait(
        InferenceRequest(
            request_id="req-1",
            model_name="echo",
            payload={"text": "hello"},
            caller="web",
        )
    )
    print(result.data)
finally:
    scheduler.stop()
```

## 核心概念

- `ModelDefinition`：模型注册信息，包含类路径、初始化参数和模块字典
- `ModelRegistry`：模型注册表，负责按名称查找模型定义
- `EmbeddedProcessPoolStrategy`：多进程执行策略，负责把请求发到 worker 进程
- `TaskScheduler`：任务调度器，负责并发控制、状态查询和超时处理
- `InferenceRequest` / `InferenceResult`：请求和结果模型
- `PlatformError`：统一异常类型，带有错误码 `ErrorCode`

## 配置

### 调度器

`SchedulerConfig` 支持以下配置：

- `max_outstanding_tasks`：最多同时允许多少个任务处于待处理或运行中
- `num_workers`：调度 worker 线程数量
- `max_retained_terminal_tasks`：保留多少个已完成/失败任务记录，`0` 表示不自动淘汰

### Worker 组

`WorkerGroup` 用于描述一个 worker 组：

- `name`：组名
- `device`：设备标识，例如 `cpu` 或 `cuda:0`
- `process_count`：该组启动多少个进程
- `models`：这个组可服务的模型列表，留空时表示服务注册表里的全部模型
- `environment`：附加给 worker 进程的环境变量
- `safety`：worker 故障后的处理策略

`safety.mode` 目前支持：

- `degrade`：默认模式，worker 异常退出后继续运行，但容量会下降
- `restart`：worker 异常退出后重启
- `shutdown`：worker 异常退出后关闭整个池

### 日志

推荐在程序启动时先调用 `configure_logging()`，再创建运行时对象。

```python
from infly.runtime.log import configure_logging

configure_logging(
    log_root="logs/infly",
    log_level="INFO",
    save_days=30,
    log_format="%(asctime)s - %(name)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s",
)
```

不调用时会使用内置默认值：`logs/infly`、`INFO`、`30`、`%(asctime)s - %(name)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s`。

#### 主线程

主线程通常只负责初始化和注册日志 sink。

```python
from infly.runtime.log import configure_logging
from infly.runtime.strategy.embedded_process_pool import EmbeddedProcessPoolStrategy

configure_logging(log_level="INFO")

pool = EmbeddedProcessPoolStrategy(registry, worker_groups)

def on_log(record):
    print(record.levelname, record.getMessage())

pool.log_manager.add_sink(on_log)
```

#### 多线程

多线程场景里，每个线程直接获取 logger 即可，不需要单独再初始化一次。

```python
from infly.runtime.log import get_logger, log_context

log = get_logger("worker")

def run_task(task_id: str) -> None:
    with log_context("task", task_id):
        log.info("task started")
```

#### 多进程

如果你自己写子进程入口，在子进程启动后先安装队列日志，再创建 logger：

```python
from infly.runtime.log import get_logger, setup_worker_logging

def worker_main(log_queue) -> None:
    setup_worker_logging(log_queue)
    log = get_logger("worker")
    log.info("worker started")
```

如果你使用的是 `EmbeddedProcessPoolStrategy`，这一步已经由 runtime 自动完成，子进程不用再手动调用 `setup_worker_logging()`。

## 错误处理

所有运行时错误都会抛出 `PlatformError`，并携带 `ErrorCode`。常见场景包括：

- `MODEL_NOT_FOUND`：模型没有注册
- `NOT_FOUND`：任务或记录不存在
- `OVERLOADED`：任务超过并发上限
- `TIMEOUT`：等待任务超时
- `WORKER_UNAVAILABLE`：worker 不可用
- `INVALID_INPUT`：参数不合法
- `INTERNAL_ERROR`：内部错误

## 测试

```bash
pytest -q
```

如果你想运行带开发依赖的测试环境：

```bash
pip install -e ".[dev]"
pytest -q
```

## 目录概览

- `infly/core`：请求、结果、错误码、协议与模型定义
- `infly/runtime`：调度器、注册表、模型加载、日志和执行策略
- `tests`：单元测试和集成测试

## TODO

- [ ] 用日志提升系统观测性
- [ ] 提升跨进程图片传输速度
- [ ] 增加对模型推理的观测，包含推理时间
