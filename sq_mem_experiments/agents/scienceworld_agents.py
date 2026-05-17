"""ScienceWorld agents: heuristic base policy + all SQ-Mem variants.

The base policy scores candidate actions by keyword overlap with the task
goal and recent observations — no LLM required.  The SQ-Mem agent layers
the memory Q-value estimator on top of that base score.
"""
import random
import re
from dataclasses import dataclass
from typing import Any

from sq_mem_experiments.agents.base import BaseAgent
from sq_mem_experiments.agents.llm_policy import LLMBasePolicy
from sq_mem_experiments.memory.compiler import compile_observation_only, compile_raw, compile_scienceworld
from sq_mem_experiments.memory.embeddings import Embedder
from sq_mem_experiments.memory.soft_q_memory import SoftQMemory
from sq_mem_experiments.schema import CandidateAction, Decision, MemoryItem


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    lambda_memory: float = 1.0
    rho_uncertainty: float = 0.5
    use_returns: bool = True              # False → semantic_retrieval variant
    action_conditioning: bool = True      # False → state-only retrieval (V(s) estimator)
    structured_state: bool = True         # False + raw_observation_only=False → raw_prefix_sq_mem (interleaved transcript)
    raw_observation_only: bool = False    # True (with structured_state=False) → just last obs, no history
    uncertainty_penalty: bool = True      # False → sq_mem_no_uncertainty
    weight_mode: str = "softmax"      # "top1" / "uniform" / "softmax"
    top_r: int = 10
    alpha: float = 0.5
    beta: float = 0.1
    epsilon: float = 0.0                  # ε-greedy exploration probability at test time


# ---------------------------------------------------------------------------
# Heuristic base policy helpers
# ---------------------------------------------------------------------------

_ACTION_KEYWORDS = {
    "boil": ["boil", "heat", "stove", "fire", "water", "temperature"],
    "melt": ["melt", "heat", "stove", "fire", "temperature"],
    "use-thermometer": ["thermometer", "measure", "temperature", "read"],
    "freeze": ["freeze", "cool", "cold", "ice"],
    "mix": ["mix", "combine", "stir", "pour"],
    "default": ["pick", "put", "move", "examine", "use", "open", "close"],
}


def heuristic_base_score(action: str, task_goal: str, observation: str) -> float:
    """Simple keyword-overlap score; gives non-uniform but deterministic priors."""
    text = action.lower()
    goal_tokens = set(re.findall(r"\b\w+\b", task_goal.lower()))
    obs_tokens = set(re.findall(r"\b\w+\b", observation.lower()[-300:]))
    action_tokens = set(re.findall(r"\b\w+\b", text))

    goal_overlap = len(action_tokens & goal_tokens) / (len(goal_tokens) + 1)
    obs_overlap = len(action_tokens & obs_tokens) / (len(obs_tokens) + 1)

    # Small bonus for task-specific keywords
    task_key = next(
        (k for k in _ACTION_KEYWORDS if k in task_goal.lower()), "default"
    )
    kw_bonus = sum(
        0.1 for kw in _ACTION_KEYWORDS[task_key] if kw in text
    )
    return goal_overlap + 0.3 * obs_overlap + kw_bonus


# ---------------------------------------------------------------------------
# Raw-history / no-memory baseline
# ---------------------------------------------------------------------------

class ScienceWorldBaseAgent(BaseAgent):
    """Heuristic base policy with no external memory."""

    def __init__(self, llm_policy: LLMBasePolicy | None = None, epsilon: float = 0.0) -> None:
        super().__init__("raw_history")
        self._llm_policy = llm_policy
        self._epsilon = epsilon
        self._rng = random.Random()
        self._task_id = ""
        self._task_goal = ""
        self._last_obs = ""
        self._actions: list[str] = []
        self._decisions: list[Decision] = []
        self._step = 0

    def reset(self, task_id: str, task_goal: str) -> None:
        self._task_id = task_id
        self._task_goal = task_goal
        self._last_obs = ""
        self._actions = []
        self._decisions = []
        self._step = 0
        # Seed per task so exploration is comparable across variants on the same task
        self._rng = random.Random(hash((task_id, self.variant, "explore")) & 0xFFFFFFFF)

    def act(self, observation: str, valid_actions: list[str]) -> str:
        self._last_obs = observation
        if not valid_actions:
            return "look around"
        if self._llm_policy:
            scores = self._llm_policy.score_actions(
                observation, valid_actions, self._task_goal, self._actions
            )
        else:
            scores = {a: heuristic_base_score(a, self._task_goal, observation) for a in valid_actions}
        argmax_selected = max(scores, key=lambda a: scores[a])
        if self._epsilon > 0 and self._rng.random() < self._epsilon:
            selected = self._rng.choice(valid_actions)
        else:
            selected = argmax_selected

        candidates = [
            CandidateAction(
                action_text=a,
                action_name=a,
                base_score=scores[a],
                combined_score=scores[a],
            )
            for a in valid_actions
        ]
        d = Decision(
            task_id=self._task_id,
            step_index=self._step,
            state_text=observation[:500],
            observation=observation,
            candidates=candidates,
            base_selected_action=argmax_selected,
            selected_action=selected,
            memory_changed_decision=False,
        )
        self._decisions.append(d)
        self._actions.append(selected)
        self._step += 1
        return selected

    def pop_decisions(self) -> list[Decision]:
        out, self._decisions = self._decisions, []
        return out


# ---------------------------------------------------------------------------
# Summary-memory agent (current-prefix compression, no prior-episode retrieval)
# ---------------------------------------------------------------------------

class ScienceWorldSummaryMemAgent(BaseAgent):
    """Uses a compressed summary of the current trajectory as context."""

    def __init__(self, llm_policy: LLMBasePolicy | None = None, epsilon: float = 0.0) -> None:
        super().__init__("summary_memory")
        self._llm_policy = llm_policy
        self._epsilon = epsilon
        self._rng = random.Random()
        self._task_goal = ""
        self._task_id = ""
        self._observations: list[str] = []
        self._actions: list[str] = []
        self._decisions: list[Decision] = []
        self._step = 0

    def reset(self, task_id: str, task_goal: str) -> None:
        self._task_id = task_id
        self._task_goal = task_goal
        self._observations = []
        self._actions = []
        self._decisions = []
        self._step = 0
        self._rng = random.Random(hash((task_id, self.variant, "explore")) & 0xFFFFFFFF)

    def act(self, observation: str, valid_actions: list[str]) -> str:
        self._observations.append(observation)
        if not valid_actions:
            return "look around"

        summary = compile_scienceworld(
            self._observations, self._actions, self._task_goal
        )
        if self._llm_policy:
            scores = self._llm_policy.score_actions(
                observation, valid_actions, self._task_goal, self._actions
            )
        else:
            summary_tokens = set(re.findall(r"\b\w+\b", summary.lower()))
            scores = {}
            for a in valid_actions:
                base = heuristic_base_score(a, self._task_goal, observation)
                summary_bonus = 0.2 * len(
                    set(re.findall(r"\b\w+\b", a.lower())) & summary_tokens
                )
                scores[a] = base + summary_bonus

        argmax_selected = max(scores, key=lambda a: scores[a])
        if self._epsilon > 0 and self._rng.random() < self._epsilon:
            selected = self._rng.choice(valid_actions)
        else:
            selected = argmax_selected
        candidates = [
            CandidateAction(
                action_text=a,
                action_name=a,
                base_score=scores[a],
                combined_score=scores[a],
            )
            for a in valid_actions
        ]
        d = Decision(
            task_id=self._task_id,
            step_index=self._step,
            state_text=summary,
            observation=observation,
            candidates=candidates,
            base_selected_action=argmax_selected,
            selected_action=selected,
            memory_changed_decision=False,
        )
        self._decisions.append(d)
        self._actions.append(selected)
        self._step += 1
        return selected

    def pop_decisions(self) -> list[Decision]:
        out, self._decisions = self._decisions, []
        return out


# ---------------------------------------------------------------------------
# Memory-builder agent (cheap policy for training trajectories)
# ---------------------------------------------------------------------------

class ScienceWorldRandomMemoryBuilderAgent(BaseAgent):
    """Mixes task-heuristic and random exploration for memory collection."""

    def __init__(self, heuristic_prob: float = 0.6, seed: int = 0) -> None:
        super().__init__("memory_builder")
        import random
        self._rng = random.Random(seed)
        self._heuristic_prob = heuristic_prob
        self._task_goal = ""
        self._task_id = ""
        self._observations: list[str] = []
        self._actions: list[str] = []
        self._score: float = 0.0
        self._decisions: list[Decision] = []
        self._step = 0

    def reset(self, task_id: str, task_goal: str) -> None:
        self._task_id = task_id
        self._task_goal = task_goal
        self._observations = []
        self._actions = []
        self._score = 0.0
        self._decisions = []
        self._step = 0

    def act(self, observation: str, valid_actions: list[str]) -> str:
        self._observations.append(observation)
        if not valid_actions:
            return "look around"

        if self._rng.random() < self._heuristic_prob:
            scores = {a: heuristic_base_score(a, self._task_goal, observation) for a in valid_actions}
            selected = max(scores, key=lambda a: scores[a])
        else:
            selected = self._rng.choice(valid_actions)

        base = heuristic_base_score(selected, self._task_goal, observation)
        candidates = [
            CandidateAction(action_text=a, action_name=a, base_score=base)
            for a in valid_actions
        ]
        d = Decision(
            task_id=self._task_id,
            step_index=self._step,
            state_text=compile_scienceworld(
                self._observations, self._actions, self._task_goal, self._score
            ),
            observation=observation,
            candidates=candidates,
            base_selected_action=selected,
            selected_action=selected,
            memory_changed_decision=False,
        )
        self._decisions.append(d)
        self._actions.append(selected)
        self._step += 1
        return selected

    def update(self, _reward: float, _done: bool, info: dict[str, Any]) -> None:
        self._score = float(info.get("score", self._score))

    def pop_decisions(self) -> list[Decision]:
        out, self._decisions = self._decisions, []
        return out


# ---------------------------------------------------------------------------
# Core SQ-Mem agent  (handles all memory-based variants via config)
# ---------------------------------------------------------------------------

class ScienceWorldSQMemAgent(BaseAgent):
    """
    Full Soft-Q Memory agent and all its ablations.

    variant controls what goes into the decision log; the behaviour is
    determined by AgentConfig and the (possibly transformed) memory items.
    """

    def __init__(
        self,
        variant: str,
        memory_items: list[MemoryItem],
        embedder: Embedder,
        config: AgentConfig,
        llm_policy: LLMBasePolicy | None = None,
    ) -> None:
        super().__init__(variant)
        self._config = config
        self._embedder = embedder
        self._llm_policy = llm_policy
        self._sqm = SoftQMemory(
            items=memory_items,
            embedder=embedder,
            top_r=config.top_r,
            alpha=config.alpha,
            beta=config.beta,
            action_conditioning=config.action_conditioning,
            weight_mode=config.weight_mode,
        )
        self._task_goal = ""
        self._task_id = ""
        self._observations: list[str] = []
        self._actions: list[str] = []
        self._score: float = 0.0
        self._decisions: list[Decision] = []
        self._step = 0
        self._rng = random.Random()

    def reset(self, task_id: str, task_goal: str) -> None:
        self._task_id = task_id
        self._task_goal = task_goal
        self._observations = []
        self._actions = []
        self._score = 0.0
        self._decisions = []
        self._step = 0
        self._rng = random.Random(hash((task_id, self.variant, "explore")) & 0xFFFFFFFF)

    def _compile_state(self) -> str:
        if self._config.structured_state:
            return compile_scienceworld(
                self._observations, self._actions, self._task_goal, self._score
            )
        if self._config.raw_observation_only:
            return compile_observation_only(self._observations)
        return compile_raw(self._observations, self._actions)

    def act(self, observation: str, valid_actions: list[str]) -> str:  # noqa: C901
        self._observations.append(observation)
        if not valid_actions:
            return "look around"

        state_text = self._compile_state()
        if self._llm_policy:
            base_scores = self._llm_policy.score_actions(
                observation, valid_actions, self._task_goal, self._actions
            )
        else:
            base_scores = {a: heuristic_base_score(a, self._task_goal, observation) for a in valid_actions}
        base_selected = max(base_scores, key=lambda a: base_scores[a])

        combined: dict[str, float] = {}
        mem_qs: dict[str, float] = {}
        mem_sigmas: dict[str, float] = {}
        best_retrievals: dict[str, tuple[list[str], list[str], list[float], list[float]]] = {}

        # Batch-embed state once + all candidate actions once (huge speedup on ST embedder)
        batch_results = self._sqm.estimate_batch(state_text, valid_actions)
        for action, (q, sigma, retrievals) in zip(valid_actions, batch_results):
            if not self._config.use_returns:
                # semantic_retrieval: use avg similarity score, ignore returns
                sim_bonus = float(sum(r.score for r in retrievals) / len(retrievals)) if retrievals else 0.0
                effective_q = sim_bonus
                effective_sigma = 0.0
            else:
                effective_q = q
                effective_sigma = sigma if self._config.uncertainty_penalty else 0.0

            mem_qs[action] = effective_q
            mem_sigmas[action] = effective_sigma
            combined[action] = (
                base_scores[action]
                + self._config.lambda_memory * effective_q
                - self._config.rho_uncertainty * effective_sigma
            )
            best_retrievals[action] = (
                [r.item.item_id for r in retrievals],
                [r.item.action_text for r in retrievals],
                [r.item.return_value for r in retrievals],
                [r.weight for r in retrievals],
            )

        argmax_selected = max(combined, key=lambda a: combined[a])
        if self._config.epsilon > 0 and self._rng.random() < self._config.epsilon:
            selected = self._rng.choice(valid_actions)
        else:
            selected = argmax_selected
        # memory_changed tracks reranking, not exploration:
        memory_changed = argmax_selected != base_selected

        # Build candidate list for the decision log
        candidates = [
            CandidateAction(
                action_text=a,
                action_name=a,
                base_score=base_scores[a],
                combined_score=combined[a],
                memory_q=mem_qs[a],
                memory_sigma=mem_sigmas[a],
            )
            for a in valid_actions
        ]

        ids, acts, rets, wts = best_retrievals[selected]
        d = Decision(
            task_id=self._task_id,
            step_index=self._step,
            state_text=state_text,
            observation=observation,
            candidates=candidates,
            base_selected_action=base_selected,
            selected_action=selected,
            memory_changed_decision=memory_changed,
            memory_q=mem_qs[selected],
            memory_sigma=mem_sigmas[selected],
            retrieved_memory_ids=ids,
            retrieved_actions=acts,
            retrieved_returns=rets,
            retrieval_weights=wts,
        )
        self._decisions.append(d)
        self._actions.append(selected)
        self._step += 1
        return selected

    def update(self, _reward: float, _done: bool, info: dict[str, Any]) -> None:
        self._score = float(info.get("score", self._score))

    def pop_decisions(self) -> list[Decision]:
        out, self._decisions = self._decisions, []
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

#: Maps variant name → (memory_transform, AgentConfig overrides)
_VARIANT_SPECS: dict[str, tuple[str | None, dict[str, Any]]] = {
    "raw_history":                    (None,            {}),
    "summary_memory":                 (None,            {}),
    "semantic_retrieval":             ("none",          {"use_returns": False}),
    # NOTE: state_only_value_memory and sq_mem_no_action_conditioning are mechanically
    # equivalent — both produce V(s) (Q identical across actions). We keep both as
    # a parameter-path invariance check: same outcome through alpha=1.0 vs the explicit flag.
    "state_only_value_memory":        ("none",          {"action_conditioning": False, "alpha": 1.0}),
    "sq_mem_no_action_conditioning":  ("none",          {"action_conditioning": False}),
    # raw_prefix_sq_mem and sq_mem_no_structured_state are now mechanically distinct:
    # raw_prefix uses compile_raw (interleaved transcript with action history),
    # no_structured_state uses just the current observation (no history at all).
    "raw_prefix_sq_mem":              ("none",          {"structured_state": False}),
    "sq_mem_no_structured_state":     ("none",          {"structured_state": False, "raw_observation_only": True}),
    "sq_mem":                         ("none",          {}),
    "sq_mem_no_returns":              ("zero",          {}),
    "sq_mem_no_uncertainty":          ("none",          {"uncertainty_penalty": False}),
    "sq_mem_top1":                    ("none",          {"weight_mode": "top1"}),
    "sq_mem_uniform_weights":         ("none",          {"weight_mode": "uniform"}),
    "sq_mem_random_memory":           ("random_memory", {}),
    "sq_mem_shuffled_returns":        ("shuffle",       {}),
    "sq_mem_value_reversed":          ("reverse",       {}),
    "sq_mem_zero_returns":            ("zero",          {}),
}

ALL_VARIANTS: list[str] = list(_VARIANT_SPECS.keys())


def make_agent(
    variant: str,
    memory_items: list[MemoryItem],
    embedder: Embedder,
    base_config: AgentConfig | None = None,
    llm_policy: LLMBasePolicy | None = None,
) -> "ScienceWorldBaseAgent | ScienceWorldSummaryMemAgent | ScienceWorldSQMemAgent":
    if variant not in _VARIANT_SPECS:
        raise ValueError(f"Unknown variant '{variant}'. Known: {list(_VARIANT_SPECS)}")

    eps = float(base_config.epsilon) if base_config is not None else 0.0

    if variant == "raw_history":
        return ScienceWorldBaseAgent(llm_policy=llm_policy, epsilon=eps)
    if variant == "summary_memory":
        return ScienceWorldSummaryMemAgent(llm_policy=llm_policy, epsilon=eps)

    cfg = AgentConfig(**(vars(base_config) if base_config else {}))
    _, overrides = _VARIANT_SPECS[variant]
    for k, v in overrides.items():
        setattr(cfg, k, v)

    return ScienceWorldSQMemAgent(
        variant=variant,
        memory_items=memory_items,
        embedder=embedder,
        config=cfg,
        llm_policy=llm_policy,
    )
