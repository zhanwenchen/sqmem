"""Abstract agent interface."""
from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    def __init__(self, variant: str):
        self.variant = variant

    @abstractmethod
    def reset(self, task_id: str, task_goal: str) -> None:
        """Called at the start of each episode."""

    @abstractmethod
    def act(self, observation: str, valid_actions: list[str]) -> str:
        """Select an action given the current observation and legal action set."""

    def update(self, _reward: float, _done: bool, _info: dict[str, Any]) -> None:
        """Optional post-step hook (e.g. to record reward)."""
