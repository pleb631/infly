from enum import StrEnum


class ErrorCode(StrEnum):
    OK = "OK"
    NOT_FOUND = "NOT_FOUND"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    OVERLOADED = "OVERLOADED"
    TIMEOUT = "TIMEOUT"
    WORKER_UNAVAILABLE = "WORKER_UNAVAILABLE"
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_REQUEST = "INVALID_REQUEST"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class PlatformError(Exception):
    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


__all__ = ["ErrorCode", "PlatformError"]
