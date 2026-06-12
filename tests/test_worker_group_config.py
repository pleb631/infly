import pytest
from pydantic import ValidationError

from infly.runtime.config import (
    SchedulerConfig,
    StrategyConfig,
    WorkerGroup,
    WorkerSafetyPolicy,
)


def test_worker_configuration_defaults() -> None:
    policy = WorkerSafetyPolicy()
    group = WorkerGroup(name="cpu", device="cpu")
    config = StrategyConfig(
        worker_groups=[WorkerGroup(name="cpu", device="cpu")]
    )

    assert policy.mode == "degrade"
    assert policy.restart_limit == 3
    assert policy.restart_window_seconds == 60
    assert policy.restart_backoff_seconds == 1

    assert group.process_count == 1
    assert group.models == []
    assert group.environment == {}
    assert group.safety.mode == "degrade"
    assert config.worker_groups[0].name == "cpu"
    assert config.embedded_pool_startup_timeout_seconds == 300


def test_scheduler_retention_configuration_defaults_and_accepts_zero() -> None:
    assert SchedulerConfig().max_retained_terminal_tasks == 50
    assert SchedulerConfig(max_retained_terminal_tasks=0).max_retained_terminal_tasks == 0


def test_scheduler_retention_configuration_rejects_negative_limit() -> None:
    with pytest.raises(ValidationError):
        SchedulerConfig(max_retained_terminal_tasks=-1)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: WorkerSafetyPolicy(mode="invalid"),
        lambda: WorkerSafetyPolicy(restart_limit=-1),
    ],
)
def test_worker_safety_policy_rejects_invalid_values(factory) -> None:
    with pytest.raises(ValidationError):
        factory()


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
