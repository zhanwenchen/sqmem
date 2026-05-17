"""Abstract environment interface for all benchmark adapters."""
from abc import ABC, abstractmethod
from typing import Any

from sq_mem_experiments.schema import TaskSpec


class BaseEnv(ABC):
    @abstractmethod
    def reset(self, task_spec: TaskSpec) -> str:
        """Reset for a task and return the initial observation."""

    @abstractmethod
    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        """Execute action. Returns (observation, reward, done, info)."""

    @abstractmethod
    def get_valid_actions(self) -> list[str]:
        """Return the list of legal actions in the current state."""

    @abstractmethod
    def close(self) -> None:
        """Release any held resources."""

    @property
    @abstractmethod
    def task_goal(self) -> str:
        """Human-readable task description."""
