import pytest
from pydantic import ValidationError

from infly.runtime.config import WorkerGroup


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": " ", "device": "x"},
        {"name": "x", "device": " "},
        {"name": "cpu", "device": "cpu", "models": ["echo", "echo"]},
        {"name": "cpu", "device": "cpu", "environment": {"INFLY_DEVICE": "cuda:0"}},
    ],
)
def test_worker_group_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValidationError):
        WorkerGroup(**kwargs)
