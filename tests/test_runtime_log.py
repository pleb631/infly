import logging
import time
from queue import Queue

import infly.runtime.log as runtime_log
from infly.runtime.log import (
    ContextFilter,
    RoutingQueueListener,
    log_context,
)


def _record() -> logging.LogRecord:
    return logging.LogRecord(
        name="infly",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )


def test_log_context_applies_and_restores_defaults() -> None:
    filt = ContextFilter()

    record = _record()
    assert filt.filter(record) is True
    assert record.log_category == ""
    assert record.log_name == "infly"

    with log_context("task", "job-1"):
        contextual = _record()
        assert filt.filter(contextual) is True
        assert contextual.log_category == "task"
        assert contextual.log_name == "job-1"

    restored = _record()
    assert filt.filter(restored) is True
    assert restored.log_category == ""
    assert restored.log_name == "infly"


def test_routing_queue_listener_routes_records_by_category_and_name(monkeypatch) -> None:
    class FakeHandler:
        def __init__(self) -> None:
            self.level = logging.INFO
            self.records: list[logging.LogRecord] = []
            self.closed = False

        def handle(self, record: logging.LogRecord) -> None:
            self.records.append(record)

        def close(self) -> None:
            self.closed = True

    file_handlers: dict[tuple[str, str], FakeHandler] = {}

    def build_file_handler(category: str, name: str, settings) -> FakeHandler:
        handler = FakeHandler()
        file_handlers[(category, name)] = handler
        return handler

    console_handler = FakeHandler()
    monkeypatch.setattr(runtime_log, "_build_file_handler", build_file_handler)
    monkeypatch.setattr(
        runtime_log,
        "_build_console_handler",
        lambda *args, **kwargs: console_handler,
    )

    listener = RoutingQueueListener(Queue())
    listener.start()
    try:
        first = _record()
        first.log_category = "task"
        first.log_name = "job-1"
        listener.handle(first)

        second = _record()
        second.log_category = "task"
        second.log_name = "job-1"
        listener.handle(second)

        third = _record()
        third.log_category = "task"
        third.log_name = "job-2"
        listener.handle(third)
    finally:
        listener.stop()

    assert list(file_handlers) == [("task", "job-1"), ("task", "job-2")]
    assert [record.msg for record in file_handlers[("task", "job-1")].records] == [
        "hello",
        "hello",
    ]
    assert [record.msg for record in file_handlers[("task", "job-2")].records] == [
        "hello"
    ]
    assert len(console_handler.records) == 3


def test_routing_queue_listener_can_customize_routing_and_handlers(monkeypatch) -> None:
    class FakeHandler:
        def __init__(self) -> None:
            self.level = logging.INFO
            self.records: list[logging.LogRecord] = []
            self.closed = False

        def handle(self, record: logging.LogRecord) -> None:
            self.records.append(record)

        def close(self) -> None:
            self.closed = True

    file_handlers: dict[str, FakeHandler] = {}

    def routing_key_factory(record: logging.LogRecord) -> str:
        return f"{record.log_category}:{record.log_name}:{record.process}:{record.thread}"

    def file_handler_factory(record: logging.LogRecord) -> FakeHandler:
        key = routing_key_factory(record)
        handler = FakeHandler()
        file_handlers[key] = handler
        return handler

    console_handler = FakeHandler()
    monkeypatch.setattr(
        runtime_log,
        "_build_console_handler",
        lambda *args, **kwargs: console_handler,
    )

    listener = RoutingQueueListener(
        Queue(),
        file_handler_factory=file_handler_factory,
        routing_key_factory=routing_key_factory,
    )
    listener.start()
    try:
        first = _record()
        first.log_category = "worker"
        first.log_name = "gpu_R0"
        first.process = 101
        first.thread = 201
        listener.handle(first)

        second = _record()
        second.log_category = "worker"
        second.log_name = "gpu_R0"
        second.process = 102
        second.thread = 202
        listener.handle(second)
    finally:
        listener.stop()

    assert list(file_handlers) == [
        "worker:gpu_R0:101:201",
        "worker:gpu_R0:102:202",
    ]
    assert [record.process for record in file_handlers["worker:gpu_R0:101:201"].records] == [
        101
    ]
    assert [record.thread for record in file_handlers["worker:gpu_R0:102:202"].records] == [
        202
    ]
    assert len(console_handler.records) == 2


def test_routing_queue_listener_stop_stops_consuming(monkeypatch) -> None:
    class FakeHandler:
        def __init__(self) -> None:
            self.level = logging.INFO
            self.records: list[logging.LogRecord] = []
            self.closed = False

        def handle(self, record: logging.LogRecord) -> None:
            self.records.append(record)

        def close(self) -> None:
            self.closed = True

    file_handler = FakeHandler()
    console_handler = FakeHandler()
    monkeypatch.setattr(
        runtime_log,
        "_build_file_handler",
        lambda category, name, settings: file_handler,
    )
    monkeypatch.setattr(
        runtime_log,
        "_build_console_handler",
        lambda *args, **kwargs: console_handler,
    )

    listener = RoutingQueueListener(Queue())
    listener.start()
    try:
        first = _record()
        first.log_category = "task"
        first.log_name = "job-1"
        listener.queue.put(first)

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and len(file_handler.records) < 1:
            time.sleep(0.01)

        assert len(file_handler.records) == 1
        listener.stop()

        second = _record()
        second.log_category = "task"
        second.log_name = "job-1"
        listener.queue.put(second)
        time.sleep(0.2)
    finally:
        if listener.thread.is_alive():
            listener.stop()

    assert not listener.thread.is_alive()
    assert len(file_handler.records) == 1
    assert len(console_handler.records) == 1
    assert file_handler.closed is True
    assert console_handler.closed is True


def test_setup_main_logging_installs_queue_handler_and_cleans_up_component_logger(
    monkeypatch,
) -> None:
    class FakeQueue:
        def __init__(self) -> None:
            self.closed = False
            self.joined = False

        def close(self) -> None:
            self.closed = True

        def join_thread(self) -> None:
            self.joined = True

    class FakeContext:
        def __init__(self) -> None:
            self.calls: list[int] = []
            self.queue = FakeQueue()

        def Queue(self, maxsize: int) -> FakeQueue:
            self.calls.append(maxsize)
            return self.queue

    component_logger = logging.getLogger("infly")
    original_handlers = list(component_logger.handlers)
    original_level = component_logger.level
    original_propagate = component_logger.propagate

    context = FakeContext()
    manager = runtime_log.setup_main_logging(mp_context=context)
    try:
        assert context.calls == [-1]
        assert manager.queue is context.queue
        assert any(
            isinstance(handler, runtime_log.QueueHandler)
            for handler in component_logger.handlers
        )
        assert component_logger.level == logging.DEBUG
        assert component_logger.propagate is False
        manager.stop()
        assert not any(
            isinstance(handler, runtime_log.QueueHandler)
            for handler in component_logger.handlers
        )
        assert component_logger.level == logging.NOTSET
        assert component_logger.propagate is True
        assert context.queue.closed is True
        assert context.queue.joined is True
    finally:
        component_logger.handlers = original_handlers
        component_logger.setLevel(original_level)
        component_logger.propagate = original_propagate


def test_routing_queue_listener_respects_runtime_log_level_changes(
    monkeypatch,
) -> None:
    class FakeHandler:
        def __init__(self) -> None:
            self.level = logging.NOTSET
            self.records: list[logging.LogRecord] = []

        def handle(self, record: logging.LogRecord) -> None:
            self.records.append(record)

        def close(self) -> None:
            pass

    old_settings = runtime_log._logging_settings()
    file_handler = FakeHandler()
    console_handler = FakeHandler()
    captured: list[str] = []

    monkeypatch.setattr(
        runtime_log,
        "_build_file_handler",
        lambda category, name, settings: file_handler,
    )
    monkeypatch.setattr(
        runtime_log,
        "_build_console_handler",
        lambda *args, **kwargs: console_handler,
    )

    runtime_log.configure_logging(
        log_root=old_settings.log_root,
        log_level=logging.DEBUG,
        save_days=old_settings.save_days,
        log_format=old_settings.log_format,
    )

    listener = RoutingQueueListener(Queue())
    listener.add_sink(lambda record: captured.append(record.getMessage()))
    try:
        info = _record()
        listener.handle(info)
        assert len(file_handler.records) == 1
        assert len(console_handler.records) == 1
        assert captured == ["hello"]

        runtime_log.configure_logging(
            log_root=old_settings.log_root,
            log_level=logging.ERROR,
            save_days=old_settings.save_days,
            log_format=old_settings.log_format,
        )

        suppressed = _record()
        listener.handle(suppressed)
        assert len(file_handler.records) == 1
        assert len(console_handler.records) == 1
        assert captured == ["hello"]

        error = logging.LogRecord(
            name="infly",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="boom",
            args=(),
            exc_info=None,
        )
        error.log_category = "task"
        error.log_name = "job-1"
        listener.handle(error)
        assert [record.msg for record in file_handler.records] == ["hello", "boom"]
        assert [record.msg for record in console_handler.records] == [
            "hello",
            "boom",
        ]
        assert captured == ["hello", "boom"]
    finally:
        listener.stop()
        runtime_log.configure_logging(
            log_root=old_settings.log_root,
            log_level=old_settings.log_level,
            save_days=old_settings.save_days,
            log_format=old_settings.log_format,
        )
