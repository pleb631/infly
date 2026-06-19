import pytest

from infly.runtime.config import (
    SchedulerConfig,
    StrategyConfig,
    WorkerGroup,
    WorkerSafetyPolicy,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": " ", "device": "x"},
        {"name": "x", "device": " "},
        {"name": "cpu", "device": "cpu", "handlers": ["echo", "echo"]},
        {"name": "cpu", "device": "cpu", "environment": {"INFLY_DEVICE": "cuda:0"}},
    ],
)
def test_worker_group_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        WorkerGroup(**kwargs)


def test_worker_group_coerces_nested_safety_mapping() -> None:
    group = WorkerGroup(
        name="cpu",
        device="cpu",
        safety={"mode": "restart", "restart_limit": 5},
    )

    assert isinstance(group.safety, WorkerSafetyPolicy)
    assert group.safety.mode == "restart"
    assert group.safety.restart_limit == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_outstanding_tasks": 0},
        {"num_workers": 0},
        {"max_retained_terminal_tasks": -1},
    ],
)
def test_scheduler_config_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        SchedulerConfig(**kwargs)


def test_strategy_config_coerces_worker_group_mappings() -> None:
    config = StrategyConfig(
        worker_groups=[
            {
                "name": "cpu",
                "device": "cpu",
                "safety": {"mode": "restart"},
            }
        ]
    )

    assert len(config.worker_groups) == 1
    assert isinstance(config.worker_groups[0], WorkerGroup)
    assert config.worker_groups[0].safety.mode == "restart"


def test_strategy_config_rejects_non_positive_startup_timeout() -> None:
    with pytest.raises(ValueError):
        StrategyConfig(embedded_pool_startup_timeout_seconds=0)
