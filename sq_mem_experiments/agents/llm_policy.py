"""LLM-based base policy.

Supports two providers:
  - "anthropic"  — Anthropic API (requires ANTHROPIC_API_KEY)
  - "local"      — any OpenAI-compatible server (Ollama, LM Studio,
                   llama.cpp, vLLM …) via the `openai` package

Config example (scienceworld_smoke.yaml):

  agent:
    llm:
      enabled: true
      provider: local           # or "anthropic"
      url: http://localhost:11434/v1   # Ollama default
      model: qwen2.5:3b
      max_candidates: 20

Prompt caching is applied for the Anthropic provider. For Haiku 4.5
the minimum cacheable prefix is 2048 tokens; the marker is harmless
if the prompt is shorter and kicks in automatically when it grows.
"""
import re
from typing import Any

_SYSTEM_TEXT = (
    "You are an agent in a text-based simulation game. "
    "Your only job is to pick the single best next action toward "
    "completing the task described.\n\n"
    "Important strategies:\n"
    "- Read the task description carefully. Identify the SPECIFIC objects "
    "named in the task (e.g. 'pencil' is not 'pen', 'mug' is not 'cup').\n"
    "- Avoid repeating an action you just took unless the world state has "
    "materially changed since then. If examining a receptacle revealed no "
    "useful objects, MOVE to a different receptacle rather than examining "
    "it again.\n"
    "- If you've taken the same kind of action 2-3 times with no reward, "
    "try a different category of action (move/take/open/close/etc.).\n\n"
    "Reply with ONLY a single integer — the 1-indexed number of your "
    "chosen action from the numbered list. No explanation. Just the integer."
)


def _build_user_content(
    task_goal: str,
    observation: str,
    recent_actions: list[str],
    candidates: list[str],
    memory_context: list[dict[str, Any]] | None = None,
    per_action_q: dict[str, tuple[float, float]] | None = None,
    state_value_prior: float | None = None,
) -> str:
    recent_for_avoid = recent_actions[-5:]
    if recent_for_avoid:
        history_block = (
            f"Recent actions you just took (DO NOT repeat any of these "
            f"unless the situation has materially changed):\n"
            + "\n".join(f"  - {a}" for a in recent_for_avoid)
        )
    else:
        history_block = "Recent actions: none yet — this is the start of the task."

    # Memory-as-context section (RAG mode). Each example surfaces an action
    # taken in a similar past state and the outcome-score that followed.
    # The LLM is expected to *abstract* over instance-specific tokens (e.g.
    # "shelf 1" → "shelf") when applying these patterns to the current state.
    memory_block = ""
    if memory_context:
        lines: list[str] = []
        for ex in memory_context:
            act = ex.get("action_text", "")
            ret = ex.get("return_value", None)
            if ret is None:
                lines.append(f"  - {act!r}")
            else:
                lines.append(f"  - {act!r} → outcome score {float(ret):.2f}")
        memory_block = (
            "Examples from similar past situations (these are PATTERNS, not "
            "literal actions — adapt the action shape to your current state):\n"
            + "\n".join(lines)
            + "\n\n"
        )

    # State-value prior (variant B): soft-aggregated return over states like
    # this one. A single scalar hint — informs the LLM whether the current
    # state has historically been close to a good outcome.
    prior_block = ""
    if state_value_prior is not None:
        prior_block = (
            f"Memory prior: similar past states reached an average final "
            f"outcome score of {float(state_value_prior):.2f}.\n\n"
        )

    # Per-action memory Q annotations (variant C): each candidate gets its
    # soft-Q estimate and uncertainty. The LLM weighs them itself — no
    # arithmetic combine. Higher σ means less reliable.
    if per_action_q is not None:
        numbered_lines: list[str] = []
        for i, a in enumerate(candidates):
            anno = per_action_q.get(a)
            if anno is None:
                numbered_lines.append(f"{i + 1}. {a}")
            else:
                q, sigma = anno
                numbered_lines.append(
                    f"{i + 1}. {a}    [memory Q={q:.2f}, σ={sigma:.2f}]"
                )
        numbered = "\n".join(numbered_lines)
        q_note = (
            "Each action's [memory Q, σ] reflects soft-aggregated returns from "
            "similar past situations. Higher Q is better; higher σ means less "
            "confident. Use these as evidence, not as commands.\n\n"
        )
    else:
        numbered = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(candidates))
        q_note = ""

    return (
        f"Task: {task_goal}\n\n"
        f"{history_block}\n\n"
        f"{memory_block}"
        f"{prior_block}"
        f"Current observation:\n{observation[:800]}\n\n"
        f"{q_note}"
        f"Valid actions:\n{numbered}\n\n"
        "Reply with the number of the single best next action."
    )


def _parse_index(text: str, n_candidates: int) -> int:
    m = re.search(r"\d+", text.strip())
    if m:
        idx = int(m.group()) - 1
        if 0 <= idx < n_candidates:
            return idx
    return 0


class LLMBasePolicy:
    """Score candidate actions using a language model.

    Works with the Anthropic API or any OpenAI-compatible local server.
    Falls back silently to heuristic rank-1 on any API error.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        max_candidates: int = 20,
        provider: str = "anthropic",
        url: str = "http://localhost:11434/v1",
        max_tokens: int | None = None,  # None → 8 for anthropic, 64 for local
        repetition_window: int = 3,
    ) -> None:
        self.model = model
        self.max_candidates = max_candidates
        self.provider = provider
        self.repetition_window = max(0, int(repetition_window))
        self._client: Any = None

        self._max_tokens = max_tokens if max_tokens is not None else (8 if provider == "anthropic" else 64)

        if provider == "anthropic":
            try:
                import anthropic as _ant
            except ImportError as exc:
                raise ImportError(
                    "anthropic package required. Run: pip install anthropic"
                ) from exc
            self._client = _ant.Anthropic()
            self._system_blocks: list[dict[str, str | dict[str, str]]] = [
                {
                    "type": "text",
                    "text": _SYSTEM_TEXT,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif provider == "local":
            try:
                from openai import OpenAI  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "openai package required for local provider. "
                    "Run: pip install openai"
                ) from exc
            self._client = OpenAI(base_url=url, api_key="local")
        else:
            raise ValueError(f"Unknown provider '{provider}'. Use 'anthropic' or 'local'.")

    def score_actions(
        self,
        observation: str,
        valid_actions: list[str],
        task_goal: str,
        recent_actions: list[str],
        memory_context: list[dict[str, Any]] | None = None,
        per_action_q: dict[str, tuple[float, float]] | None = None,
        state_value_prior: float | None = None,
    ) -> dict[str, float]:
        """Return a score in [0, 1] for each valid action.

        The selected action gets 1.0. Other pre-filtered candidates get a
        small graded score so memory Q-values can still override when the
        evidence is strong. Actions outside the pre-filtered set get 0.0.

        When `memory_context` is provided (RAG-context mode), each example
        is rendered into the prompt as an "in similar past situations…"
        section so the LLM can reason over it. The LLM remains the decider;
        no Q-value reranking happens downstream in RAG mode.
        """
        from sq_mem_experiments.agents.scienceworld_agents import heuristic_base_score as _base_score  # local import avoids circular dep

        ranked = sorted(
            valid_actions,
            key=lambda a: _base_score(a, task_goal, observation),
            reverse=True,
        )
        # Demote actions taken in the last `repetition_window` steps — this
        # breaks the small-LLM "examine shelf 1 × 7" trap without any
        # benchmark-specific code. We don't *delete* the repeated actions
        # (the agent might genuinely need to repeat one); we just push them
        # to the end of the candidate list so the LLM rarely picks them, and
        # SQ-Mem's Q-value can still rerank them in if memory says they help.
        if self.repetition_window > 0 and recent_actions:
            recent_set = set(recent_actions[-self.repetition_window:])
            non_repeated = [a for a in ranked if a not in recent_set]
            repeated = [a for a in ranked if a in recent_set]
            ranked = non_repeated + repeated
        candidates = ranked[: self.max_candidates]
        user_content = _build_user_content(
            task_goal,
            observation,
            recent_actions,
            candidates,
            memory_context,
            per_action_q=per_action_q,
            state_value_prior=state_value_prior,
        )

        selected_idx = 0
        try:
            if self.provider == "anthropic":
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=self._max_tokens,
                    system=self._system_blocks,  # type: ignore[arg-type]
                    messages=[{"role": "user", "content": user_content}],
                )
                text = resp.content[0].text  # type: ignore[union-attr]
            else:  # local OpenAI-compatible
                # /no_think disables Qwen3 chain-of-thought; harmless on other models
                resp = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=self._max_tokens,
                    messages=[
                        {"role": "system", "content": _SYSTEM_TEXT},
                        {"role": "user", "content": user_content + " /no_think"},
                    ],
                    temperature=0,
                )
                text = resp.choices[0].message.content or ""
            selected_idx = _parse_index(text, len(candidates))
        except Exception:  # noqa: BLE001 — intentional broad catch; any API failure → heuristic fallback
            pass

        scores: dict[str, float] = {a: 0.0 for a in valid_actions}
        for i, a in enumerate(candidates):
            scores[a] = (self.max_candidates - i) / (self.max_candidates * 20)
        scores[candidates[selected_idx]] = 1.0
        return scores
