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
    "You are an agent in a text-based science simulation. "
    "Your only job is to pick the single best next action. "
    "Reply with ONLY a single integer — the 1-indexed number "
    "of your chosen action. No explanation. No reasoning. "
    "Just the integer."
)


def _build_user_content(
    task_goal: str,
    observation: str,
    recent_actions: list[str],
    candidates: list[str],
) -> str:
    history = " → ".join(recent_actions[-5:]) if recent_actions else "none"
    numbered = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(candidates))
    return (
        f"Task: {task_goal}\n\n"
        f"Recent actions: {history}\n\n"
        f"Current observation:\n{observation[:800]}\n\n"
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
    ) -> None:
        self.model = model
        self.max_candidates = max_candidates
        self.provider = provider
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
    ) -> dict[str, float]:
        """Return a score in [0, 1] for each valid action.

        The selected action gets 1.0. Other pre-filtered candidates get a
        small graded score so memory Q-values can still override when the
        evidence is strong. Actions outside the pre-filtered set get 0.0.
        """
        from sq_mem_experiments.agents.scienceworld_agents import heuristic_base_score as _base_score  # local import avoids circular dep

        ranked = sorted(
            valid_actions,
            key=lambda a: _base_score(a, task_goal, observation),
            reverse=True,
        )
        candidates = ranked[: self.max_candidates]
        user_content = _build_user_content(task_goal, observation, recent_actions, candidates)

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
