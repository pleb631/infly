"""Read-only health reporting for process-pool workers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from infly.runtime.config import WorkerGroup
from infly.runtime.observability import HealthStatus, StrategyHealthSnapshot
from infly.runtime.strategy.state import PoolSnapshot
from infly.runtime.strategy.worker import WorkerState


class ProcessPoolHealthReporter:
    """Build a health snapshot from the pool's current worker state."""

    @staticmethod
    def snapshot(
        *,
        name: str,
        groups: Mapping[str, WorkerGroup],
        state: PoolSnapshot,
    ) -> StrategyHealthSnapshot:
        workers = state.workers
        total_workers = len(workers)
        alive_workers = 0
        alive_by_group: dict[str, int] = defaultdict(int)
        for worker in workers:
            if not ProcessPoolHealthReporter._is_alive(worker):
                continue
            alive_workers += 1
            alive_by_group[worker.group.name] += 1
        restarting_workers = sum(1 for worker in workers if worker.next_restart_at is not None)
        degraded_workers = total_workers - alive_workers - restarting_workers

        if total_workers == 0 or state.close_complete or (not state.accepting and alive_workers == 0):
            status = HealthStatus.DOWN
        elif alive_workers == total_workers:
            status = HealthStatus.OK
        else:
            status = HealthStatus.DEGRADED

        group_detail: dict[str, dict[str, int | bool]] = {}
        for group_name, group in groups.items():
            group_detail[group_name] = {
                "configured_processes": group.process_count,
                "alive_workers": alive_by_group[group_name],
                "accepting": state.accepting,
            }

        return StrategyHealthSnapshot(
            name=name,
            status=status,
            accepting=state.accepting,
            detail={
                "total_workers": total_workers,
                "alive_workers": alive_workers,
                "restarting_workers": restarting_workers,
                "degraded_workers": max(degraded_workers, 0),
                "groups": group_detail,
            },
        )

    @staticmethod
    def _is_alive(worker: WorkerState) -> bool:
        return worker.is_running()
