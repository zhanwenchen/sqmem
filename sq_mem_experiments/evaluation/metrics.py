"""Aggregate metrics computed from a list of episodes."""
from __future__ import annotations

from typing import Any

import numpy as np

from sq_mem_experiments.schema import Decision, Episode


# ---------------------------------------------------------------------------
# Episode-level flags
# ---------------------------------------------------------------------------

def has_repeated_failure(decisions: list[Decision], window: int = 3) -> bool:
    """True if the same action appears >= window times consecutively."""
    if len(decisions) < window:
        return False
    actions = [d.selected_action for d in decisions]
    for i in range(len(actions) - window + 1):
        if len(set(actions[i : i + window])) == 1:
            return True
    return False


def has_error_recovery(decisions: list[Decision]) -> bool:
    """True if the episode contains at least one failed step followed by progress."""
    rewards = [d.reward for d in decisions]
    saw_negative = False
    for r in rewards:
        if r < 0:
            saw_negative = True
        elif r > 0 and saw_negative:
            return True
    return False


# ---------------------------------------------------------------------------
# Variant-level summary
# ---------------------------------------------------------------------------

def compute_summary(episodes: list[Episode]) -> dict[str, float]:
    if not episodes:
        return {}
    success_rate = float(np.mean([e.success for e in episodes]))
    avg_steps = float(np.mean([e.steps for e in episodes]))
    total_reward = float(np.mean([e.total_reward for e in episodes]))
    repeated_failure_rate = float(
        np.mean([has_repeated_failure(e.decisions) for e in episodes])
    )
    error_recovery_rate = float(
        np.mean([has_error_recovery(e.decisions) for e in episodes])
    )
    intervention_episode_rate = float(
        np.mean([
            any(d.memory_changed_decision for d in e.decisions)
            for e in episodes
        ])
    )
    return {
        "n_episodes": float(len(episodes)),
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_total_reward": total_reward,
        "repeated_failure_rate": repeated_failure_rate,
        "error_recovery_rate": error_recovery_rate,
        "intervention_episode_rate": intervention_episode_rate,
    }


def compute_intervention_summary(
    episodes: list[Episode],
    baseline_episodes: list[Episode] | None = None,
) -> dict[str, float]:
    """Proportion of steps/episodes where memory changed the decision."""
    all_decisions = [d for e in episodes for d in e.decisions]
    if not all_decisions:
        return {}

    changed = [d for d in all_decisions if d.memory_changed_decision]
    intervention_rate = len(changed) / len(all_decisions)
    intervention_episode_rate = float(
        np.mean([any(d.memory_changed_decision for d in e.decisions) for e in episodes])
    )

    # Beneficial / harmful: compare episode success to baseline if available
    beneficial_rate = 0.0
    harmful_rate = 0.0
    if baseline_episodes:
        baseline_success = {e.task_id: e.success for e in baseline_episodes}
        changed_episodes = [
            e for e in episodes if any(d.memory_changed_decision for d in e.decisions)
        ]
        if changed_episodes:
            beneficial = sum(
                1
                for e in changed_episodes
                if e.success and not baseline_success.get(e.task_id, True)
            )
            harmful = sum(
                1
                for e in changed_episodes
                if not e.success and baseline_success.get(e.task_id, False)
            )
            total = len(changed_episodes)
            beneficial_rate = beneficial / total
            harmful_rate = harmful / total

    return {
        "intervention_rate": intervention_rate,
        "intervention_episode_rate": intervention_episode_rate,
        "beneficial_rate": beneficial_rate,
        "harmful_rate": harmful_rate,
        "n_changed_decisions": float(len(changed)),
        "n_decisions": float(len(all_decisions)),
    }


def compute_calibration(
    episodes: list[Episode], n_bins: int = 5
) -> list[dict[str, float]]:
    """Bin decisions by memory Q-value; compute empirical success rate per bin."""
    rows: list[tuple[float, bool]] = []
    for ep in episodes:
        for d in ep.decisions:
            rows.append((d.memory_q, ep.success))
    if not rows:
        return []

    qs = np.array([r[0] for r in rows])
    successes = np.array([float(r[1]) for r in rows])

    bin_edges = np.percentile(qs, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return []

    result: list[dict[str, float]] = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (qs >= lo) & (qs <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        result.append({
            "q_bin_low": float(lo),
            "q_bin_high": float(hi),
            "q_bin_center": float((lo + hi) / 2),
            "n_decisions": float(n),
            "success_rate": float(successes[mask].mean()),
        })
    return result


def compute_horizon_buckets(
    episodes: list[Episode],
) -> list[dict[str, Any]]:
    """Stratify success rate by episode length."""
    if not episodes:
        return []
    buckets = [(0, 10, "short"), (11, 20, "medium"), (21, 10**9, "long")]
    rows: list[dict[str, Any]] = []
    for lo, hi, label in buckets:
        subset = [e for e in episodes if lo <= e.steps <= hi]
        if subset:
            rows.append({
                "horizon_bucket": label,
                "min_steps": lo,
                "max_steps": hi if hi < 10**9 else -1,
                "n_episodes": len(subset),
                "success_rate": float(np.mean([e.success for e in subset])),
            })
    return rows


def bootstrap_ci(
    a_successes: list[bool],
    b_successes: list[bool],
    n_samples: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for success-rate difference (A - B).

    Returns (delta, ci_low, ci_high).
    """
    a = np.array(a_successes, dtype=float)
    b = np.array(b_successes, dtype=float)
    delta = float(a.mean() - b.mean())
    rng = np.random.RandomState(seed)
    diffs: list[float] = []
    n = len(a)
    for _ in range(n_samples):
        idx = rng.randint(0, n, size=n)
        diffs.append(float(a[idx].mean() - b[idx].mean()))
    diffs_arr = np.array(diffs, dtype=np.float64)
    ci_low = float(np.percentile(diffs_arr, 2.5))
    ci_high = float(np.percentile(diffs_arr, 97.5))
    return delta, ci_low, ci_high


def episodes_to_decisions_rows(
    episodes: list[Episode], variant: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ep in episodes:
        for d in ep.decisions:
            row = d.to_flat_dict()
            row["variant"] = variant
            row["episode_success"] = ep.success
            rows.append(row)
    return rows
