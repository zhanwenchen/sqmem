"""ScienceWorld environment adapter.

Requires: pip install scienceworld  (and Java >= 8 on PATH)
"""
import importlib.util
from typing import Any

from sq_mem_experiments.envs.base import BaseEnv
from sq_mem_experiments.schema import TaskSpec

_sw_available: bool = importlib.util.find_spec("scienceworld") is not None


class ScienceWorldAdapter(BaseEnv):
    def __init__(self, reward_mode: str = "score_delta", generate_gold_path: bool = False):
        if not _sw_available:
            raise ImportError(
                "scienceworld is not installed. "
                "Run: pip install 'sq_mem_experiments[scienceworld]'"
            )
        import scienceworld as _sw  # type: ignore[import]
        from typing import cast

        sw: Any = cast(Any, _sw)
        self.reward_mode = reward_mode
        self._generate_gold_path = generate_gold_path
        self._env: Any = sw.ScienceWorldEnv("")
        self._last_score: float = 0.0
        self._goal: str = ""
        self._gold_sequence: list[str] = []

    def reset(self, task_spec: TaskSpec) -> str:
        self._last_score = 0.0
        self._gold_sequence = []
        self._env.load(
            task_spec.task_name,
            task_spec.variation_id,
            generateGoldPath=self._generate_gold_path,
        )
        obs, info = self._env.reset()
        self._goal = str(info.get("taskDesc", "")) or str(self._env.get_task_description())
        if self._generate_gold_path:
            self._gold_sequence = list(self._env.get_gold_action_sequence())
        return str(obs)

    @property
    def gold_sequence(self) -> list[str]:
        return self._gold_sequence

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        obs, reward, done, info = self._env.step(action)
        info = dict(info)
        score: float = float(info.get("score", 0))
        if self.reward_mode == "score_delta":
            step_reward = (score - self._last_score) / 100.0
        else:
            step_reward = float(reward)
        self._last_score = score
        info["normalized_score"] = score / 100.0
        return str(obs), step_reward, bool(done), info

    def get_valid_actions(self) -> list[str]:
        # Grounded action-object combinations (e.g. "open door to kitchen") —
        # NOT get_possible_actions(), which returns templates with "OBJ" placeholders
        # that ScienceWorld does not actually accept as commands.
        return list(self._env.get_valid_action_object_combinations())

    def close(self) -> None:
        self._env.close()

    @property
    def task_goal(self) -> str:
        return self._goal

    @property
    def current_score(self) -> float:
        return self._last_score

    @staticmethod
    def make_tasks(
        task_names: list[str],
        train_variations: list[int],
        test_variations: list[int],
    ) -> tuple[list[TaskSpec], list[TaskSpec]]:
        train_tasks: list[TaskSpec] = []
        test_tasks: list[TaskSpec] = []
        for name in task_names:
            for var in train_variations:
                train_tasks.append(
                    TaskSpec(
                        task_id=f"train_{name}_var{var:04d}",
                        env_name="scienceworld",
                        task_name=name,
                        variation_id=var,
                        split="train",
                    )
                )
            for var in test_variations:
                test_tasks.append(
                    TaskSpec(
                        task_id=f"test_{name}_var{var:04d}",
                        env_name="scienceworld",
                        task_name=name,
                        variation_id=var,
                        split="test",
                    )
                )
        return train_tasks, test_tasks
