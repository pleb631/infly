from __future__ import annotations

import logging
from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from logging.handlers import QueueHandler, TimedRotatingFileHandler
from multiprocessing import Queue
from pathlib import Path
from queue import Empty
from threading import Lock, Thread
from typing import Any, Callable

DEFAULT_LOG_ROOT = Path("logs/infly")
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_SAVE_DAYS = 30
DEFAULT_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s"
)
APP_LOGGER_NAME = "infly"

_current_log_category = ContextVar("log_category", default="")
_current_log_name = ContextVar("log_name", default=APP_LOGGER_NAME)

FileHandlerFactory = Callable[[logging.LogRecord], logging.Handler]
ConsoleHandlerFactory = Callable[[], logging.Handler]
RoutingKeyFactory = Callable[[logging.LogRecord], str]
LogLevelValue = str | int
LogRecordSink = Callable[[logging.LogRecord], None]


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    log_root: Path = DEFAULT_LOG_ROOT
    log_level: int = DEFAULT_LOG_LEVEL
    save_days: int = DEFAULT_SAVE_DAYS
    log_format: str = DEFAULT_LOG_FORMAT

    def __post_init__(self) -> None:
        object.__setattr__(self, "log_root", Path(self.log_root))
        object.__setattr__(self, "log_level", _resolve_log_level(self.log_level))
        object.__setattr__(self, "save_days", int(self.save_days))

_configured_logging_settings: LoggingSettings | None = None


def _resolve_log_level(level_name: str | int) -> int:
    if isinstance(level_name, int):
        return level_name
    candidate = level_name.strip()
    if candidate.isdigit():
        return int(candidate)
    level = logging.getLevelName(candidate.upper())
    if isinstance(level, int):
        return level
    return logging.INFO


def _logging_settings() -> LoggingSettings:
    return _configured_logging_settings or LoggingSettings()


def _coerce_logging_settings(
    *,
    log_root: str | Path | None = None,
    log_level: LogLevelValue | None = None,
    save_days: int | None = None,
    log_format: str | None = None,
) -> LoggingSettings:
    current = _logging_settings()
    return LoggingSettings(
        log_root=Path(log_root) if log_root is not None else current.log_root,
        log_level=(
            _resolve_log_level(log_level)
            if isinstance(log_level, str)
            else log_level
            if log_level is not None
            else current.log_level
        ),
        save_days=save_days if save_days is not None else current.save_days,
        log_format=log_format if log_format is not None else current.log_format,
    )


def _create_main_log_queue(mp_context: Any | None):
    if mp_context is not None:
        return mp_context.Queue(-1)
    return Queue()


def configure_logging(
    *,
    log_root: str | Path | None = None,
    log_level: LogLevelValue | None = None,
    save_days: int | None = None,
    log_format: str | None = None,
) -> LoggingSettings:
    settings = _coerce_logging_settings(
        log_root=log_root,
        log_level=log_level,
        save_days=save_days,
        log_format=log_format,
    )
    global _configured_logging_settings
    _configured_logging_settings = settings
    return settings


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "log_category"):
            record.log_category = _current_log_category.get()
        if not hasattr(record, "log_name"):
            record.log_name = _current_log_name.get()
        return True


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[1mDEBUG\033[0m",
        "INFO": "\033[37mINFO\033[0m",
        "WARNING": "\033[33mWARNING\033[0m",
        "ERROR": "\033[31mERROR\033[0m",
        "CRITICAL": "\033[35mCRITICAL\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        original = record.levelname
        record.levelname = self.COLORS.get(original, original)
        try:
            return super().format(record)
        finally:
            record.levelname = original


class CategoryLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        current_category = _current_log_category.get()
        current_name = _current_log_name.get()
        if self.extra:
            if "log_category" not in extra and current_category == "":
                extra["log_category"] = self.extra.get("log_category", "")
            if "log_name" not in extra and current_name == APP_LOGGER_NAME:
                extra["log_name"] = self.extra.get("log_name", APP_LOGGER_NAME)
        return msg, kwargs


@contextmanager
def log_context(category: str = "", name: str = APP_LOGGER_NAME):
    category_token = _current_log_category.set(category)
    name_token = _current_log_name.set(name)
    try:
        yield
    finally:
        _current_log_name.reset(name_token)
        _current_log_category.reset(category_token)


def _safe_filename(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(":", "_")

def _install_queue_handler(
    log_queue: Queue,
    settings: LoggingSettings,
) -> tuple[logging.Logger, list[logging.Handler], int, bool]:
    app_logger = logging.getLogger(APP_LOGGER_NAME)

    previous_handlers = list(app_logger.handlers)
    previous_level = app_logger.level
    previous_propagate = app_logger.propagate

    app_logger.handlers.clear()
    app_logger.setLevel(logging.DEBUG)
    app_logger.propagate = False

    queue_handler = QueueHandler(log_queue)
    queue_handler.setLevel(settings.log_level)
    queue_handler.addFilter(ContextFilter())
    app_logger.addHandler(queue_handler)

    return app_logger, previous_handlers, previous_level, previous_propagate


def _build_file_handler(category: str, name: str, settings: LoggingSettings):
    log_dir = settings.log_root / category
    log_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_filename(name)
    log_file = log_dir / f"{safe_name}.log"

    handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        interval=1,
        backupCount=settings.save_days,
        encoding="utf8",
        delay=True,
        utc=True,
    )
    handler.setLevel(settings.log_level)
    handler.addFilter(ContextFilter())
    handler.setFormatter(logging.Formatter(settings.log_format))
    return handler


def _build_console_handler(settings: LoggingSettings, color: bool = True):
    handler = logging.StreamHandler()
    handler.setLevel(settings.log_level)
    handler.addFilter(ContextFilter())
    if color:
        handler.setFormatter(ColorFormatter(settings.log_format))
    else:
        handler.setFormatter(logging.Formatter(settings.log_format))
    return handler


class RoutingQueueListener:
    def __init__(
        self,
        queue: Queue,
        *,
        settings: LoggingSettings | None = None,
        file_handler_factory: FileHandlerFactory | None = None,
        console_handler_factory: ConsoleHandlerFactory | None = None,
        routing_key_factory: RoutingKeyFactory | None = None,
    ):
        self.queue = queue
        self._settings = settings
        self.handlers: dict[str, logging.Handler] = {}
        self._sinks: list[LogRecordSink] = []
        self.lock = Lock()
        self.thread = Thread(target=self._run, daemon=True, name="InflyLogListener")
        self._file_handler_factory = (
            file_handler_factory or self._default_file_handler_factory
        )
        self._routing_key_factory = (
            routing_key_factory or self._default_routing_key_factory
        )
        self.console_handler = (
            console_handler_factory
            or (lambda: _build_console_handler(self._effective_settings(), color=True))
        )()

    def _effective_settings(self) -> LoggingSettings:
        return self._settings or _logging_settings()

    def start(self):
        self.thread.start()

    def stop(self):
        if self.thread.is_alive():
            self.queue.put(None)
            self.thread.join(timeout=5)

        with self.lock:
            for handler in self.handlers.values():
                handler.close()
            self.handlers.clear()

        self.console_handler.close()

    @staticmethod
    def _default_routing_key_factory(record: logging.LogRecord) -> str:
        category = getattr(record, "log_category", "")
        name = getattr(record, "log_name", APP_LOGGER_NAME)
        return f"{category}:{name}"

    def _default_file_handler_factory(self, record: logging.LogRecord):
        category = getattr(record, "log_category", "")
        name = getattr(record, "log_name", APP_LOGGER_NAME)
        return _build_file_handler(category, name, self._effective_settings())

    def _get_file_handler(self, record: logging.LogRecord):
        key = self._routing_key_factory(record)
        with self.lock:
            if key in self.handlers:
                return self.handlers[key]
            handler = self._file_handler_factory(record)
            self.handlers[key] = handler
            return handler

    def handle(self, record: logging.LogRecord):
        if record.levelno < self._effective_settings().log_level:
            return
        file_handler = self._get_file_handler(record)

        if record.levelno >= file_handler.level:
            file_handler.handle(record)
        if record.levelno >= self.console_handler.level:
            self.console_handler.handle(record)
            self._notify_sinks(record)

    def add_sink(self, sink: LogRecordSink) -> None:
        with self.lock:
            self._sinks.append(sink)

    def remove_sink(self, sink: LogRecordSink) -> None:
        with self.lock:
            self._sinks = [candidate for candidate in self._sinks if candidate is not sink]

    def _notify_sinks(self, record: logging.LogRecord) -> None:
        with self.lock:
            sinks = list(self._sinks)
        for sink in sinks:
            try:
                sink(record)
            except Exception:
                continue

    def _run(self):
        while True:
            try:
                record = self.queue.get(timeout=0.2)
            except Empty:
                continue
            if record is None:
                break
            self.handle(record)


class MainLogManager:
    def __init__(
        self,
        logger: logging.Logger,
        queue: Queue,
        listener: RoutingQueueListener,
        *,
        settings: LoggingSettings,
        root_handlers: list[logging.Handler],
        root_level: int,
        root_propagate: bool,
    ):
        self._logger = logger
        self.queue = queue
        self.listener = listener
        self.settings = settings
        self._root_handlers = root_handlers
        self._root_level = root_level
        self._root_propagate = root_propagate
        self._started = False
        self._stopped = False
        self._lock = Lock()

    def start(self) -> None:
        with self._lock:
            if self._started or self._stopped:
                return
            self.listener.start()
            self._started = True

    def add_sink(self, sink: LogRecordSink) -> None:
        self.listener.add_sink(sink)

    def remove_sink(self, sink: LogRecordSink) -> None:
        self.listener.remove_sink(sink)

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
        try:
            if self._started:
                self.listener.stop()
        finally:
            self._logger.handlers = list(self._root_handlers)
            self._logger.setLevel(logging.NOTSET)
            self._logger.propagate = self._root_propagate
            close = getattr(self.queue, "close", None)
            if close is not None:
                close()
            join_thread = getattr(self.queue, "join_thread", None)
            if join_thread is not None:
                join_thread()


def setup_main_logging(
    *,
    settings: LoggingSettings | None = None,
    mp_context: Any | None = None,
    file_handler_factory: FileHandlerFactory | None = None,
    console_handler_factory: ConsoleHandlerFactory | None = None,
    routing_key_factory: RoutingKeyFactory | None = None,
    start: bool = False,
) -> MainLogManager:
    effective_settings = settings or _logging_settings()
    log_queue = _create_main_log_queue(mp_context)
    app_logger, root_handlers, root_level, root_propagate = _install_queue_handler(
        log_queue, effective_settings
    )
    listener = RoutingQueueListener(
        log_queue,
        settings=effective_settings,
        file_handler_factory=file_handler_factory,
        console_handler_factory=console_handler_factory,
        routing_key_factory=routing_key_factory,
    )
    manager = MainLogManager(
        app_logger,
        log_queue,
        listener,
        settings=effective_settings,
        root_handlers=root_handlers,
        root_level=root_level,
        root_propagate=root_propagate,
    )
    if start:
        manager.start()
    return manager


def setup_worker_logging(
    log_queue: Queue,
    *,
    settings: LoggingSettings | None = None,
):
    effective_settings = settings or _logging_settings()
    _install_queue_handler(log_queue, effective_settings)


def get_logger(name: str = APP_LOGGER_NAME, category: str = ""):
    normalized_name = name or APP_LOGGER_NAME
    logger_name = (
        APP_LOGGER_NAME
        if normalized_name == APP_LOGGER_NAME
        else f"{APP_LOGGER_NAME}.{normalized_name}"
    )
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    return CategoryLoggerAdapter(
        logger,
        {
            "log_category": category,
            "log_name": normalized_name,
        },
    )


__all__ = [
    "ColorFormatter",
    "ContextFilter",
    "MainLogManager",
    "LoggingSettings",
    "RoutingQueueListener",
    "CategoryLoggerAdapter",
    "LogRecordSink",
    "configure_logging",
    "get_logger",
    "log_context",
    "setup_main_logging",
    "setup_worker_logging",
]
