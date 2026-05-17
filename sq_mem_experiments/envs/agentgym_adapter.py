"""AgentGym environment adapter (stub — not yet implemented)."""
from typing import Any

from sq_mem_experiments.envs.base import BaseEnv
from sq_mem_experiments.schema import TaskSpec


class AgentGymAdapter(BaseEnv):
    def reset(self, task_spec: TaskSpec) -> str:
        raise NotImplementedError("AgentGym adapter not yet implemented")

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        raise NotImplementedError

    def get_valid_actions(self) -> list[str]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    @property
    def task_goal(self) -> str:
        return ""
