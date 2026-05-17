"""State-text compilers: turn a trajectory prefix into a structured string."""
from typing import Any


def compile_raw(observations: list[str], actions: list[str]) -> str:
    """Last-N interleaved observations and actions, unstructured."""
    pairs = list(zip(observations, actions))[-10:]
    parts = [f"Step {i}: {obs}\nAction: {act}" for i, (obs, act) in enumerate(pairs)]
    if len(observations) > len(actions):
        parts.append(f"Step {len(actions)}: {observations[-1]}")
    return "\n".join(parts)


def compile_observation_only(observations: list[str]) -> str:
    """Just the current observation — no history, no task goal, no score."""
    return observations[-1] if observations else ""


def compile_summary(
    observations: list[str],
    actions: list[str],
    task_goal: str = "",
) -> str:
    """Structured summary of the current trajectory prefix (no prior episodes)."""
    lines: list[str] = []
    if task_goal:
        lines.append(f"Goal: {task_goal}")
    lines.append(f"Step: {len(actions)}")
    if observations:
        lines.append(f"Current observation: {observations[-1][:400]}")
    if actions:
        n = min(5, len(actions))
        lines.append(f"Last {n} actions: {' -> '.join(actions[-n:])}")
    return "\n".join(lines)


def compile_scienceworld(
    observations: list[str],
    actions: list[str],
    task_goal: str = "",
    score: float = 0.0,
) -> str:
    """ScienceWorld-specific structured prefix state."""
    lines: list[str] = []
    if task_goal:
        lines.append(f"Task: {task_goal}")
    lines.append(f"Score: {score:.3f}")
    lines.append(f"Steps: {len(actions)}")
    if observations:
        lines.append(f"Observation: {observations[-1][:500]}")
    if actions:
        lines.append(f"Last action: {actions[-1]}")
    if len(actions) >= 3:
        lines.append(f"Recent actions: {' | '.join(actions[-3:])}")
    return "\n".join(lines)


def compile_generic(
    observations: list[str],
    actions: list[str],
    task_goal: str = "",
    metadata: dict[str, Any] | None = None,
) -> str:
    """General-purpose structured prefix state for non-ScienceWorld envs."""
    meta = metadata or {}
    lines: list[str] = []
    if task_goal:
        lines.append(f"Goal: {task_goal}")
    lines.append(f"Steps completed: {len(actions)}")
    if observations:
        lines.append(f"Current state: {observations[-1][:400]}")
    if actions:
        n = min(5, len(actions))
        lines.append(f"Last {n} actions: {' -> '.join(actions[-n:])}")
    for k, v in meta.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)
