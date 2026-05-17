import json
from dataclasses import asdict, dataclass, field
from typing import Any


def _str_list() -> list[str]:
    return []


def _float_list() -> list[float]:
    return []


def _any_dict() -> dict[str, Any]:
    return {}


@dataclass
class CandidateAction:
    action_text: str
    action_name: str
    base_score: float = 0.0
    combined_score: float = 0.0
    memory_q: float = 0.0
    memory_sigma: float = 0.0


@dataclass
class Decision:
    task_id: str
    step_index: int
    state_text: str
    observation: str
    candidates: list[CandidateAction]
    base_selected_action: str
    selected_action: str
    memory_changed_decision: bool
    reward: float = 0.0
    return_value: float = 0.0
    memory_q: float = 0.0
    memory_sigma: float = 0.0
    retrieved_memory_ids: list[str] = field(default_factory=_str_list)
    retrieved_actions: list[str] = field(default_factory=_str_list)
    retrieved_returns: list[float] = field(default_factory=_float_list)
    retrieval_weights: list[float] = field(default_factory=_float_list)

    def to_flat_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step_index": self.step_index,
            "base_selected_action": self.base_selected_action,
            "selected_action": self.selected_action,
            "memory_changed_decision": self.memory_changed_decision,
            "reward": self.reward,
            "return_value": self.return_value,
            "memory_q": self.memory_q,
            "memory_sigma": self.memory_sigma,
            "retrieved_memory_ids_json": json.dumps(self.retrieved_memory_ids),
            "retrieved_actions_json": json.dumps(self.retrieved_actions),
            "selected_retrieved_returns_json": json.dumps(self.retrieved_returns),
            "retrieval_weights_json": json.dumps(self.retrieval_weights),
            "n_candidates": len(self.candidates),
        }


@dataclass
class Episode:
    task_id: str
    variant: str
    decisions: list[Decision]
    success: bool
    total_reward: float
    steps: int
    metadata: dict[str, Any] = field(default_factory=_any_dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("decisions")
        return d


@dataclass
class TaskSpec:
    task_id: str
    env_name: str
    task_name: str
    variation_id: int
    split: str
    metadata: dict[str, Any] = field(default_factory=_any_dict)


@dataclass
class MemoryItem:
    item_id: str
    task_id: str
    split: str
    step_index: int
    state_text: str
    action_text: str
    action_name: str
    return_value: float
    state_vec: list[float]
    action_vec: list[float]
    metadata: dict[str, Any] = field(default_factory=_any_dict)
