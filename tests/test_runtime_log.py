import logging
import time
from pathlib import Path
from queue import Queue

import infly.runtime.log as runtime_log
from infly.runtime.log import (
    ContextFilter,
    RoutingQueueListener,
    _safe_filename,
    get_logger,
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
    assert record.log_name == "sys"

    with log_context("task", "job-1"):
        contextual = _record()
        assert filt.filter(contextual) is True
        assert contextual.log_category == "task"
        assert contextual.log_name == "job-1"

    restored = _record()
    assert filt.filter(restored) is True
    assert restored.log_category == ""
    assert restored.log_name == "sys"


def test_safe_filename_sanitizes_path_separators() -> None:
    assert _safe_filename("a/b\\c:d") == "a_b_c_d"


def test_build_file_handler_uses_current_environment(monkeypatch) -> None:
    first_root = Path.cwd() / ".codex" / "runtime-log-test" / "first-root"
    second_root = Path.cwd() / ".codex" / "runtime-log-test" / "second-root"

    monkeypatch.setenv("INFLY_LOG_ROOT", str(first_root))
    monkeypatch.setenv("INFLY_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("INFLY_LOG_SAVE_DAYS", "7")
    monkeypatch.setenv("INFLY_LOG_FORMAT", "first %(message)s")

    first = runtime_log._build_file_handler("task", "job-1")
    try:
        assert Path(first.baseFilename).parent == first_root / "task"
        assert first.level == logging.DEBUG
        assert first.backupCount == 7
        assert first.formatter._fmt == "first %(message)s"
    finally:
        first.close()

    monkeypatch.setenv("INFLY_LOG_ROOT", str(second_root))
    monkeypatch.setenv("INFLY_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("INFLY_LOG_SAVE_DAYS", "3")
    monkeypatch.setenv("INFLY_LOG_FORMAT", "second %(message)s")

    second = runtime_log._build_file_handler("task", "job-2")
    try:
        assert Path(second.baseFilename).parent == second_root / "task"
        assert second.level == logging.WARNING
        assert second.backupCount == 3
        assert second.formatter._fmt == "second %(message)s"
    finally:
        second.close()


def test_get_logger_reuses_direct_cache_and_preserves_context_fields() -> None:
    runtime_log._direct_logger_cache.clear()
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    try:
        root.handlers = []

        logger1 = runtime_log.get_logger("worker", "task")
        logger2 = runtime_log.get_logger("worker", "task")

        assert logger1 is logger2
        assert logger1.extra["log_category"] == "task"
        assert logger1.extra["log_name"] == "worker"
    finally:
        root.handlers = original_handlers
        runtime_log._direct_logger_cache.clear()


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

    def build_file_handler(category: str, name: str) -> FakeHandler:
        handler = FakeHandler()
        file_handlers[(category, name)] = handler
        return handler

    console_handler = FakeHandler()
    monkeypatch.setattr(runtime_log, "_build_file_handler", build_file_handler)
    monkeypatch.setattr(runtime_log, "_build_console_handler", lambda color=True: console_handler)

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
    monkeypatch.setattr(runtime_log, "_build_file_handler", lambda category, name: file_handler)
    monkeypatch.setattr(runtime_log, "_build_console_handler", lambda color=True: console_handler)

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


def test_get_logger_uses_queue_mode_when_root_has_queue_handler(monkeypatch) -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    queue_handler = runtime_log.QueueHandler(Queue())
    try:
        root.handlers = [queue_handler]
        logger = get_logger("worker", "task")

        assert logger.extra["log_category"] == "task"
        assert logger.extra["log_name"] == "worker"
        assert logger.logger.propagate is True
    finally:
        root.handlers = original_handlers
