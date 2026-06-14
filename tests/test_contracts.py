from infly.core.errors import ErrorCode, PlatformError


def test_platform_error_exposes_code_and_message() -> None:
    error = PlatformError(ErrorCode.NOT_FOUND, "not found")
    assert error.code == ErrorCode.NOT_FOUND
    assert error.message == "not found"
    assert str(error) == "not found"
