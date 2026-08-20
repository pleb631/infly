import pytest

from infly.runtime.config import (
    SchedulerConfig,
    WorkerGroup,
    WorkerSafetyPolicy,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": " "},
        {"name": "cpu", "handlers": ["echo", "echo"]},
    ],
)
def test_worker_group_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        WorkerGroup(**kwargs)


def test_worker_group_coerces_nested_safety_mapping() -> None:
    group = WorkerGroup(
        name="cpu",
        safety={"mode": "restart", "restart_limit": 5},
    )

    assert isinstance(group.safety, WorkerSafetyPolicy)
    assert group.safety.mode == "restart"
    assert group.safety.restart_limit == 5


def test_worker_group_does_not_accept_or_reserve_device_configuration() -> None:
    group = WorkerGroup(
        name="cpu",
        environment={"INFLY_DEVICE": "cuda:0"},
    )

    assert group.environment == {"INFLY_DEVICE": "cuda:0"}
    with pytest.raises(TypeError, match="device"):
        WorkerGroup(name="cpu", device="cpu")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_outstanding_tasks": 0},
        {"num_threads": 0},
        {"max_retained_terminal_tasks": -1},
    ],
)
def test_scheduler_config_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        SchedulerConfig(**kwargs)
