"""Worker-group validation and routing metadata for process pools."""

from __future__ import annotations

from collections import defaultdict

from infly.core.errors import ErrorCode, PlatformError
from infly.runtime.config import WorkerGroup
from infly.runtime.registry import HandlerRegistry
from infly.runtime.strategy.worker import WorkerState


class WorkerGroupCatalog:
    """Own the configured groups and their handler-to-group index."""

    def __init__(self, registry: HandlerRegistry) -> None:
        self._registry = registry
        self.groups: dict[str, WorkerGroup] = {}
        self.group_handlers: dict[str, tuple[str, ...]] = {}
        self.handler_groups: dict[str, list[str]] = defaultdict(list)

    def configure_initial(self, groups: list[WorkerGroup]) -> list[WorkerState]:
        if not groups:
            raise PlatformError(ErrorCode.INTERNAL_ERROR, "ProcessPoolStrategy requires at least one worker group")
        names = [group.name for group in groups]
        if len(names) != len(set(names)):
            raise PlatformError(ErrorCode.INTERNAL_ERROR, "WorkerGroup names must be unique")

        prepared = [(group, self.resolve_handlers(group)) for group in groups]
        workers: list[WorkerState] = []
        for group, handler_names in prepared:
            self.publish(group, handler_names)
            workers.extend(self.create_workers(group, handler_names))
        return workers

    def resolve_handlers(self, group: WorkerGroup) -> tuple[str, ...]:
        all_handlers = tuple(definition.handler_name for definition in self._registry.list())
        handler_names = tuple(group.handlers) if group.handlers else all_handlers
        for handler_name in handler_names:
            try:
                self._registry.get(handler_name)
            except PlatformError as exc:
                raise PlatformError(
                    ErrorCode.INTERNAL_ERROR,
                    f"WorkerGroup '{group.name}' references missing handler '{handler_name}'",
                ) from exc
        return handler_names

    def ensure_can_register(self, group_name: str, *, closed: bool) -> None:
        if closed:
            raise PlatformError(ErrorCode.INTERNAL_ERROR, "ProcessPoolStrategy is closed")
        if group_name in self.groups:
            raise PlatformError(ErrorCode.INTERNAL_ERROR, f"WorkerGroup '{group_name}' is already registered")

    def publish(self, group: WorkerGroup, handler_names: tuple[str, ...]) -> None:
        self.groups[group.name] = group
        self.group_handlers[group.name] = handler_names
        for handler_name in handler_names:
            self.handler_groups[handler_name].append(group.name)

    def unpublish(self, group_name: str) -> WorkerGroup:
        group = self.groups.pop(group_name, None)
        if group is None:
            raise PlatformError(ErrorCode.NOT_FOUND, f"WorkerGroup '{group_name}' is not registered")
        for handler_name in self.group_handlers.pop(group_name):
            groups = self.handler_groups[handler_name]
            groups.remove(group_name)
            if not groups:
                self.handler_groups.pop(handler_name, None)
        return group

    @staticmethod
    def create_workers(group: WorkerGroup, handler_names: tuple[str, ...]) -> list[WorkerState]:
        return [
            WorkerState(
                worker_id=f"{group.name}_R{index}",
                index=index,
                group=group,
                handler_names=handler_names,
            )
            for index in range(group.process_count)
        ]
