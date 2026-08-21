"""In-memory routing policy for live process-pool workers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from infly.runtime.config import WorkerGroup
from infly.runtime.strategy.worker import WorkerState


class ProcessPoolRouter:
    """Choose a group by live capacity, then its least-loaded worker.

    Worker ownership belongs to ``PoolLifecycleState``. This class only keeps
    scheduling cursors and weights, making it a routing policy rather than a
    second worker registry.
    """

    def __init__(self, groups: Mapping[str, WorkerGroup]) -> None:
        self._groups = groups
        self._group_cursors: dict[str, int] = defaultdict(int)
        self._smooth_weights: dict[str, int] = defaultdict(int)
        self._group_order: dict[str, int] = {}
        self._known_group_names: tuple[str, ...] = ()

    @staticmethod
    def live_workers_by_group(
        workers: Iterable[WorkerState],
    ) -> dict[str, list[WorkerState]]:
        """Classify live workers once for a single dispatch decision."""
        grouped: dict[str, list[WorkerState]] = defaultdict(list)
        for worker in workers:
            if worker.is_routable():
                grouped[worker.group.name].append(worker)
        return grouped

    def select_group(
        self,
        group_names: list[str],
        live_workers_by_group: Mapping[str, list[WorkerState]],
    ) -> str:
        self._refresh_group_order()
        weights = {group_name: len(live_workers_by_group[group_name]) for group_name in group_names}
        total_weight = sum(weights.values())
        for group_name, weight in weights.items():
            self._smooth_weights[group_name] += weight
        selected = max(
            group_names,
            key=lambda group_name: (self._smooth_weights[group_name], -self._group_order[group_name]),
        )
        self._smooth_weights[selected] -= total_weight
        return selected

    def ordered_workers(
        self,
        group_name: str,
        live_workers_by_group: Mapping[str, list[WorkerState]],
    ) -> list[WorkerState]:
        live_workers = live_workers_by_group[group_name]
        cursor = self._group_cursors[group_name]
        count = self._groups[group_name].process_count
        return sorted(live_workers, key=lambda worker: (worker.outstanding, (worker.index - cursor) % count))

    def record_assignment(self, worker: WorkerState) -> None:
        self._group_cursors[worker.group.name] = (worker.index + 1) % worker.group.process_count

    def forget_group(self, group_name: str) -> None:
        self._group_cursors.pop(group_name, None)
        self._smooth_weights.pop(group_name, None)

    def reset_group_weight(self, group_name: str) -> None:
        self._smooth_weights[group_name] = 0

    def _refresh_group_order(self) -> None:
        group_names = tuple(self._groups)
        if group_names == self._known_group_names:
            return
        self._known_group_names = group_names
        self._group_order = {group_name: index for index, group_name in enumerate(group_names)}
