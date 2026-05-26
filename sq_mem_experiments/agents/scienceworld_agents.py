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
    # memory_mode: "q_rerank" → soft Q table (original SQ-Mem behaviour);
    # "rag_context" → surface retrieved (state, action, return) triples in the
    # LLM prompt and let the LLM decide. Action shapes generalize across
    # instance-specific naming (e.g. ALFWorld's "shelf 1" vs "shelf 5").
    memory_mode: str = "q_rerank"
    # How many top-K memories to surface in the LLM prompt under rag_context.
    rag_top_k: int = 5

    # Paper-faithful middle-ground flags (all compose with q_rerank mode).
    # A) normalize_actions: strip instance tokens (digits, articles) before
    #    embedding so "examine shelf 1" matches "examine shelf 5" in retrieval.
    #    Soft-Q mechanism intact; only the action key generalizes.
    normalize_actions: bool = False
    # B) state_value_in_prompt: compute V(s) via state-only soft aggregation
    #    over retrieved returns and inject it as a scalar prior in the prompt.
    state_value_in_prompt: bool = False
    # C) q_in_prompt: include per-action mem_Q and σ in the LLM prompt and
    #    let the LLM decide; arithmetic combine (score + λ·Q − ρ·σ) is skipped.
    q_in_prompt: bool = False

    # σ-gate: under q_in_prompt, if the mean σ across candidate annotations
    # exceeds this threshold, suppress the annotations entirely for that
    # decision (LLM proceeds without memory hints). Default 1.0 disables
    # gating (σ is in [0, 1], so threshold ≥ 1.0 always passes through).
    # Empirically useful values are in [0.3, 0.7] on benchmarks where memory
    # is informative when confident but misleading when uncertain.
    q_gate_sigma_threshold: float = 1.0


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
    """Heuristic base policy with no external memory.

    Despite the name, this class is also the per-rollout agent for ALFWorld's
    `self_rollout` memory builder. The state_text we store here ends up as
    the *retrieval key* in memory; test-time queries from ScienceWorldSQMemAgent
    use `compile_scienceworld(observations, actions, task_goal, score)`. To
    keep keys matchable, we compile state the same way here — otherwise the
    embedder compares a query like "Task: look at mug…" to a memory key like
    "You pick up the plate 1 from the cabinet 2." and matches on kitchen-room
    objects rather than on the task identity.
    """

    def __init__(
        self,
        llm_policy: LLMBasePolicy | None = None,
        epsilon: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__("raw_history")
        self._llm_policy = llm_policy
        self._epsilon = epsilon
        self._seed = seed
        self._rng = random.Random()
        self._task_id = ""
        self._task_goal = ""
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
        # Seed scheme: per-task by default (so exploration is comparable across
        # variants on the same test task), plus an optional `seed` salt the
        # memory-builder uses to get distinct rollouts of the same training task.
        salt: tuple[Any, ...] = (task_id, self.variant, "explore")
        if self._seed is not None:
            salt = salt + (self._seed,)
        self._rng = random.Random(hash(salt) & 0xFFFFFFFF)

    def update(self, _reward: float, _done: bool, info: dict[str, Any]) -> None:
        self._score = float(info.get("score", self._score))

    def act(self, observation: str, valid_actions: list[str]) -> str:
        self._observations.append(observation)
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
        # Compile the SAME state representation that ScienceWorldSQMemAgent uses
        # at test time — task goal at the top, recent obs/actions, current score.
        state_text = compile_scienceworld(
            self._observations, self._actions, self._task_goal, self._score
        )
        d = Decision(
            task_id=self._task_id,
            step_index=self._step,
            state_text=state_text,
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

    def __init__(
        self,
        llm_policy: LLMBasePolicy | None = None,
        epsilon: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__("summary_memory")
        self._llm_policy = llm_policy
        self._epsilon = epsilon
        self._seed = seed
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
        salt: tuple[Any, ...] = (task_id, self.variant, "explore")
        if self._seed is not None:
            salt = salt + (self._seed,)
        self._rng = random.Random(hash(salt) & 0xFFFFFFFF)

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
            normalize_actions=config.normalize_actions,
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

    def _build_rag_memory_context(self, state_text: str) -> list[dict[str, Any]]:
        """RAG mode: retrieve top-K memories by state similarity and format as
        prompt examples, applying the same variant-specific transformations
        the Q-rerank path uses.

        - `use_returns=False` (semantic_retrieval): hide return values
        - `weight_mode="top1"`: surface only the single top match
        - other variants: returns inherit the bank-level transform that the
          runner already applies via apply_return_transform (zero/shuffle/
          reverse/random_memory), so we just pass them through.
        """
        top_k = self._config.rag_top_k
        if self._config.weight_mode == "top1":
            top_k = 1
        retrievals = self._sqm.retrieve_by_state(state_text, top_k=top_k)
        out: list[dict[str, Any]] = []
        for r in retrievals:
            entry: dict[str, Any] = {"action_text": r.item.action_text}
            if self._config.use_returns:
                entry["return_value"] = float(r.item.return_value)
            out.append(entry)
        return out

    def act(self, observation: str, valid_actions: list[str]) -> str:  # noqa: C901
        self._observations.append(observation)
        if not valid_actions:
            return "look around"

        state_text = self._compile_state()

        # ──────────────── RAG-context mode ────────────────
        # Memory is surfaced as in-context examples for the LLM, not used to
        # rerank candidate actions via a Q-value. The LLM is the decider.
        if self._config.memory_mode == "rag_context" and self._llm_policy:
            return self._act_rag_context(state_text, observation, valid_actions)

        # ──────────────── Q-rerank mode (original SQ-Mem) ────────────────
        # Batch-embed state once + all candidate actions once (huge speedup on ST embedder)
        batch_results = self._sqm.estimate_batch(state_text, valid_actions)

        mem_qs: dict[str, float] = {}
        mem_sigmas: dict[str, float] = {}
        best_retrievals: dict[str, tuple[list[str], list[str], list[float], list[float]]] = {}
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
            best_retrievals[action] = (
                [r.item.item_id for r in retrievals],
                [r.item.action_text for r in retrievals],
                [r.item.return_value for r in retrievals],
                [r.weight for r in retrievals],
            )

        # State-value prior V(s): soft-aggregated returns over state-only
        # retrieval — a single scalar hint for the LLM (variant B).
        v_prior: float | None = None
        if self._config.state_value_in_prompt:
            state_retrievals = self._sqm.retrieve_by_state(
                state_text, top_k=self._config.top_r
            )
            if state_retrievals:
                v_prior = float(
                    sum(r.item.return_value * r.weight for r in state_retrievals)
                )

        # Per-action Q/σ surfaced in the prompt (variant C). When True, the
        # LLM is the decider; arithmetic combine is skipped.
        per_action_q: dict[str, tuple[float, float]] | None = None
        if self._config.q_in_prompt:
            per_action_q = {a: (mem_qs[a], mem_sigmas[a]) for a in valid_actions}

            # σ-gate (Issue #2): if the LLM-consumed Q values are uniformly
            # uncertain (high mean σ), skip injecting them for this step.
            # The LLM falls back to its own reasoning without misleading hints.
            # Default threshold 1.0 disables gating (σ is in [0,1]).
            if (per_action_q
                and self._config.q_gate_sigma_threshold < 1.0
                and len(per_action_q) > 0):
                mean_sigma = sum(s for _, s in per_action_q.values()) / len(per_action_q)
                if mean_sigma > self._config.q_gate_sigma_threshold:
                    per_action_q = None  # gated off for this decision

        # Memory-informed scoring: LLM sees Q annotations / V prior in prompt
        # (under q_in_prompt / state_value_in_prompt), or just plain prompt.
        if self._llm_policy:
            informed_scores = self._llm_policy.score_actions(
                observation, valid_actions, self._task_goal, self._actions,
                per_action_q=per_action_q,
                state_value_prior=v_prior,
            )
        else:
            informed_scores = {a: heuristic_base_score(a, self._task_goal, observation) for a in valid_actions}

        # Counterfactual base policy (Issue #1): when q_in_prompt or
        # state_value_in_prompt is on, the informed_scores were already
        # memory-influenced. To measure whether memory changed the LLM's
        # decision (H8 intervention audit), we need a SECOND LLM call with
        # no memory hints in the prompt. This doubles LLM cost only for
        # those modes; the standard linear-combine path skips it.
        needs_counterfactual = self._llm_policy is not None and (
            self._config.q_in_prompt or self._config.state_value_in_prompt
        )
        if needs_counterfactual:
            assert self._llm_policy is not None  # narrow for type checker
            base_scores = self._llm_policy.score_actions(
                observation, valid_actions, self._task_goal, self._actions,
                per_action_q=None,
                state_value_prior=None,
            )
        else:
            base_scores = informed_scores

        base_selected = max(base_scores, key=lambda a: base_scores[a])

        combined: dict[str, float] = {}
        if self._config.q_in_prompt:
            # LLM saw Q/σ in its prompt — its score IS the combined score.
            combined = dict(informed_scores)
        else:
            for action in valid_actions:
                combined[action] = (
                    informed_scores[action]
                    + self._config.lambda_memory * mem_qs[action]
                    - self._config.rho_uncertainty * mem_sigmas[action]
                )

        argmax_selected = max(combined, key=lambda a: combined[a])
        if self._config.epsilon > 0 and self._rng.random() < self._config.epsilon:
            selected = self._rng.choice(valid_actions)
        else:
            selected = argmax_selected
        # memory_changed = the informed choice differs from the no-memory choice
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

    def _act_rag_context(
        self,
        state_text: str,
        observation: str,
        valid_actions: list[str],
    ) -> str:
        """Decision in RAG-context mode.

        The LLM is given (a) the candidate actions and (b) retrieved memory as
        prompt examples. The LLM picks the action. No Q-rerank.

        We compute a base-only score (LLM call WITHOUT memory context) so the
        decision log's `memory_changed_decision` flag captures whether the
        memory examples actually changed the LLM's choice. Analogue of the
        Q-rerank path's argmax(base) vs argmax(combined).
        """
        assert self._llm_policy is not None

        v_prior: float | None = None
        if self._config.state_value_in_prompt:
            state_retrievals = self._sqm.retrieve_by_state(
                state_text, top_k=self._config.top_r
            )
            if state_retrievals:
                v_prior = float(
                    sum(r.item.return_value * r.weight for r in state_retrievals)
                )

        # Reference choice: LLM without memory context
        base_scores = self._llm_policy.score_actions(
            observation, valid_actions, self._task_goal, self._actions,
            memory_context=None,
            state_value_prior=v_prior,
        )
        base_selected = max(base_scores, key=lambda a: base_scores[a])

        # Memory-informed choice: LLM with retrieved examples in the prompt
        memory_context = self._build_rag_memory_context(state_text)
        ctx_scores = self._llm_policy.score_actions(
            observation, valid_actions, self._task_goal, self._actions,
            memory_context=memory_context,
            state_value_prior=v_prior,
        )
        argmax_selected = max(ctx_scores, key=lambda a: ctx_scores[a])

        if self._config.epsilon > 0 and self._rng.random() < self._config.epsilon:
            selected = self._rng.choice(valid_actions)
        else:
            selected = argmax_selected
        memory_changed = argmax_selected != base_selected

        ret_actions = [c.get("action_text", "") for c in memory_context]
        ret_returns = [float(c.get("return_value", 0.0)) for c in memory_context]
        mem_q = float(sum(ret_returns) / len(ret_returns)) if ret_returns else 0.0
        mem_sigma = (
            float((sum((r - mem_q) ** 2 for r in ret_returns) / len(ret_returns)) ** 0.5)
            if ret_returns else 0.0
        )

        candidates = [
            CandidateAction(
                action_text=a,
                action_name=a,
                base_score=base_scores[a],
                combined_score=ctx_scores[a],
                memory_q=mem_q,
                memory_sigma=mem_sigma,
            )
            for a in valid_actions
        ]
        d = Decision(
            task_id=self._task_id,
            step_index=self._step,
            state_text=state_text,
            observation=observation,
            candidates=candidates,
            base_selected_action=base_selected,
            selected_action=selected,
            memory_changed_decision=memory_changed,
            memory_q=mem_q,
            memory_sigma=mem_sigma,
            retrieved_memory_ids=[],
            retrieved_actions=ret_actions,
            retrieved_returns=ret_returns,
            retrieval_weights=[1.0 / len(ret_returns)] * len(ret_returns) if ret_returns else [],
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
