import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from logging.handlers import QueueHandler, TimedRotatingFileHandler
from multiprocessing import Queue
from pathlib import Path
from queue import Empty
from threading import Lock, Thread

DEFAULT_LOG_ROOT = Path("logs/infly")
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SAVE_DAYS = 30
DEFAULT_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s"
)

_current_log_category = ContextVar("log_category", default="")
_current_log_name = ContextVar("log_name", default="sys")

_direct_logger_cache: dict[str, logging.LoggerAdapter] = {}
_direct_lock = Lock()


def _resolve_log_level(level_name: str) -> int:
    level = logging.getLevelName(level_name.upper())
    if isinstance(level, int):
        return level
    return logging.INFO


def _logging_settings() -> tuple[Path, int, int, str]:
    log_root = Path(os.getenv("INFLY_LOG_ROOT", str(DEFAULT_LOG_ROOT)))
    log_level = _resolve_log_level(os.getenv("INFLY_LOG_LEVEL", DEFAULT_LOG_LEVEL))
    save_days = int(os.getenv("INFLY_LOG_SAVE_DAYS", str(DEFAULT_SAVE_DAYS)))
    log_format = os.getenv("INFLY_LOG_FORMAT", DEFAULT_LOG_FORMAT)
    return log_root, log_level, save_days, log_format


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
        if self.extra:
            extra.setdefault("log_category", self.extra.get("log_category", ""))
            extra.setdefault("log_name", self.extra.get("log_name", "sys"))
        return msg, kwargs


@contextmanager
def log_context(category: str = "", name: str = "sys"):
    category_token = _current_log_category.set(category)
    name_token = _current_log_name.set(name)
    try:
        yield
    finally:
        _current_log_name.reset(name_token)
        _current_log_category.reset(category_token)


def _safe_filename(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(":", "_")


def _build_file_handler(category: str, name: str):
    log_root, log_level, save_days, log_format = _logging_settings()
    log_dir = log_root / category
    log_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_filename(name)
    log_file = log_dir / f"{safe_name}.log"

    handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        interval=1,
        backupCount=save_days,
        encoding="utf8",
        delay=True,
        utc=True,
    )
    handler.setLevel(log_level)
    handler.addFilter(ContextFilter())
    handler.setFormatter(logging.Formatter(log_format))
    return handler


def _build_console_handler(color: bool = True):
    _, log_level, _, log_format = _logging_settings()
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.addFilter(ContextFilter())
    if color:
        handler.setFormatter(ColorFormatter(log_format))
    else:
        handler.setFormatter(logging.Formatter(log_format))
    return handler


class RoutingQueueListener:
    def __init__(self, queue: Queue):
        self.queue = queue
        self.handlers: dict[str, logging.Handler] = {}
        self.lock = Lock()
        self.thread = Thread(target=self._run, daemon=True)
        self.console_handler = _build_console_handler(color=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.queue.put(None)
        self.thread.join(timeout=5)

        with self.lock:
            for handler in self.handlers.values():
                handler.close()
            self.handlers.clear()

        self.console_handler.close()

    def _get_file_handler(self, category: str, name: str):
        key = f"{category}:{name}"
        with self.lock:
            if key in self.handlers:
                return self.handlers[key]
            handler = _build_file_handler(category, name)
            self.handlers[key] = handler
            return handler

    def handle(self, record: logging.LogRecord):
        category = getattr(record, "log_category", "")
        name = getattr(record, "log_name", "sys")
        file_handler = self._get_file_handler(category, name)

        if record.levelno >= file_handler.level:
            file_handler.handle(record)
        if record.levelno >= self.console_handler.level:
            self.console_handler.handle(record)

    def _run(self):
        while True:
            try:
                record = self.queue.get(timeout=0.2)
            except Empty:
                continue
            if record is None:
                break
            self.handle(record)


def setup_main_logging():
    log_queue = Queue(-1)
    listener = RoutingQueueListener(log_queue)
    return log_queue, listener


def setup_worker_logging(log_queue: Queue):
    root = logging.getLogger()
    root.handlers.clear()
    _, log_level, _, _ = _logging_settings()
    root.setLevel(log_level)

    queue_handler = QueueHandler(log_queue)
    queue_handler.setLevel(log_level)
    queue_handler.addFilter(ContextFilter())
    root.addHandler(queue_handler)


def get_logger(name: str = "sys", category: str = ""):
    root = logging.getLogger()
    log_root, log_level, save_days, log_format = _logging_settings()
    settings_signature = repr((str(log_root), log_level, save_days, log_format))

    is_queue_mode = any(isinstance(h, QueueHandler) for h in root.handlers)

    if is_queue_mode:
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        logger.propagate = True
        return CategoryLoggerAdapter(
            logger,
            {
                "log_category": category,
                "log_name": name,
            },
        )

    cache_key = (
        f"{category}:{name}|{settings_signature}" if category else f"{name}|{settings_signature}"
    )
    if cache_key in _direct_logger_cache:
        return _direct_logger_cache[cache_key]

    with _direct_lock:
        if cache_key in _direct_logger_cache:
            return _direct_logger_cache[cache_key]

        logger_name = f"{category}:{name}" if category else name
        logger = logging.getLogger(logger_name)
        current_signature = getattr(logger, "_infly_settings_signature", None)
        if current_signature != settings_signature or not logger.handlers:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()
            logger.addHandler(_build_file_handler(category, name))
            logger.addHandler(_build_console_handler(color=True))
            setattr(logger, "_infly_settings_signature", settings_signature)

        logger.setLevel(log_level)
        logger.propagate = False

        setattr(logger, "_infly_settings_signature", settings_signature)

        adapter = CategoryLoggerAdapter(
            logger,
            {
                "log_category": category,
                "log_name": name,
            },
        )
        _direct_logger_cache[cache_key] = adapter
        return adapter
