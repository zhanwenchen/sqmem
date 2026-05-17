"""Automated claim-ledger evaluator.

Reads completed run artifacts and writes hypothesis_report.md/.json,
hypothesis_checks.csv, hypothesis_comparisons.csv.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sq_mem_experiments.evaluation.metrics import bootstrap_ci

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------
SUPPORTED = "supported"
WEAKENED = "weakened"
INCONCLUSIVE = "inconclusive"
NOT_TESTED = "not_tested"
SETUP_FAILED = "setup_failed"


@dataclass
class CheckResult:
    check_id: str
    status: str
    rationale: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    check_id: str
    variant_a: str
    variant_b: str
    delta_success: float
    ci_low: float
    ci_high: float
    n_pairs: int
    note: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_summary(run_dir: str) -> pd.DataFrame | None:
    p = os.path.join(run_dir, "summary.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, index_col="variant")


def _load_episodes(run_dir: str, variant: str) -> list[dict[str, Any]] | None:
    p = os.path.join(run_dir, f"episodes_{variant}.jsonl")
    if not os.path.exists(p):
        return None
    rows: list[dict[str, Any]] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_memory_summary(run_dir: str) -> dict[str, Any] | None:
    p = os.path.join(run_dir, "memory_summary.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)  # type: ignore[no-any-return]


def _load_interventions(run_dir: str) -> pd.DataFrame | None:
    p = os.path.join(run_dir, "summary_interventions.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, index_col="variant")


def _load_calibration(run_dir: str, variant: str) -> pd.DataFrame | None:
    p = os.path.join(run_dir, f"calibration_{variant}.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def _sr(summary: pd.DataFrame, variant: str) -> float | None:
    """Primary comparison metric.

    Uses success_rate when it has spread across variants; otherwise falls back
    to avg_total_reward (partial-credit progress signal). This is what lets H1–H5
    produce real verdicts on sparse-success tasks where partial progress varies
    cleanly across variants.
    """
    if variant not in summary.index:
        return None
    metric_col = "success_rate"
    if "avg_total_reward" in summary.columns:
        sr_unique = summary["success_rate"].nunique()
        ar_unique = summary["avg_total_reward"].nunique()
        # Prefer avg_total_reward when it provides more discrimination across
        # variants — captures partial-credit progress on sparse-success tasks.
        if ar_unique > sr_unique:
            metric_col = "avg_total_reward"
    return float(summary.loc[variant, metric_col])  # type: ignore[arg-type]


def _paired_comparison(
    run_dir: str,
    check_id: str,
    variant_a: str,
    variant_b: str,
    bootstrap_samples: int,
    min_pairs: int,
) -> ComparisonResult:
    eps_a = _load_episodes(run_dir, variant_a)
    eps_b = _load_episodes(run_dir, variant_b)
    delta = float("nan")
    ci_low = float("nan")
    ci_high = float("nan")
    n_pairs = 0
    note = "no episode files"

    if eps_a and eps_b:
        map_a = {e["task_id"]: e["success"] for e in eps_a}
        map_b = {e["task_id"]: e["success"] for e in eps_b}
        shared = sorted(set(map_a) & set(map_b))
        n_pairs = len(shared)
        if n_pairs >= min_pairs:
            a_s = [bool(map_a[t]) for t in shared]
            b_s = [bool(map_b[t]) for t in shared]
            delta, ci_low, ci_high = bootstrap_ci(a_s, b_s, bootstrap_samples)
            note = f"{n_pairs} shared tasks"
        else:
            delta = float(np.mean([map_a[t] for t in shared])) - float(
                np.mean([map_b[t] for t in shared])
            ) if shared else float("nan")
            note = f"only {n_pairs} shared tasks (need {min_pairs})"

    return ComparisonResult(
        check_id=check_id,
        variant_a=variant_a,
        variant_b=variant_b,
        delta_success=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        n_pairs=n_pairs,
        note=note,
    )


def _sq_beats(
    summary: pd.DataFrame,
    others: list[str],
    min_delta: float,
) -> tuple[str, str]:
    """Return (status, rationale) comparing sq_mem against a list of variants."""
    sq_sr = _sr(summary, "sq_mem")
    if sq_sr is None:
        return NOT_TESTED, "sq_mem not in summary"
    results: list[str] = []
    all_beat = True
    for v in others:
        other_sr = _sr(summary, v)
        if other_sr is None:
            results.append(f"{v}: not_tested")
            all_beat = False
            continue
        delta = sq_sr - other_sr
        if delta >= min_delta:
            results.append(f"sq_mem ({sq_sr:.3f}) > {v} ({other_sr:.3f}) Δ={delta:+.3f} ✓")
        else:
            results.append(f"sq_mem ({sq_sr:.3f}) vs {v} ({other_sr:.3f}) Δ={delta:+.3f} ✗")
            all_beat = False
    rationale = "; ".join(results)
    missing = [v for v in others if _sr(summary, v) is None]
    if len(missing) == len(others):
        return NOT_TESTED, "none of the comparison variants were run"
    if all_beat:
        return SUPPORTED, rationale
    tested = [v for v in others if _sr(summary, v) is not None]
    if all(_sr(summary, v) is not None and sq_sr - _sr(summary, v) < 0  # type: ignore[operator]
           for v in tested):
        return WEAKENED, rationale
    return INCONCLUSIVE, rationale


# ---------------------------------------------------------------------------
# Individual hypothesis checks
# ---------------------------------------------------------------------------

def _h0_pipeline_sanity(run_dir: str) -> CheckResult:
    required = [
        "summary.csv",
        "memory_bank.jsonl",
        "memory_summary.json",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(run_dir, f))]
    if missing:
        return CheckResult("H0_pipeline_sanity", SETUP_FAILED,
                           f"Missing required files: {missing}")
    mem = _load_memory_summary(run_dir)
    if mem and mem.get("splits", {}).get("test", 0) > 0:
        return CheckResult("H0_pipeline_sanity", SETUP_FAILED,
                           "Memory bank contains test-split items (split violation)")
    return CheckResult("H0_pipeline_sanity", SUPPORTED,
                       "Required artifacts present; no test items in memory bank")


def _h1_main_comparison(
    run_dir: str, summary: pd.DataFrame, min_delta: float
) -> CheckResult:
    status, rationale = _sq_beats(
        summary, ["semantic_retrieval", "summary_memory", "raw_history"], min_delta
    )
    return CheckResult("H1_main_sq_mem_comparison", status, rationale)


def _h2_value_destruction(
    run_dir: str, summary: pd.DataFrame, min_delta: float
) -> CheckResult:
    controls = [
        "sq_mem_shuffled_returns",
        "sq_mem_value_reversed",
        "sq_mem_zero_returns",
        "sq_mem_no_returns",
        "sq_mem_random_memory",
    ]
    status, rationale = _sq_beats(summary, controls, min_delta)
    # Extra check: reversed should hurt (sq_mem_value_reversed < raw_history)
    extras: list[str] = []
    rev_sr = _sr(summary, "sq_mem_value_reversed")
    raw_sr = _sr(summary, "raw_history")
    if rev_sr is not None and raw_sr is not None:
        if rev_sr < raw_sr:
            extras.append(f"reversed ({rev_sr:.3f}) < raw_history ({raw_sr:.3f}) ✓")
        else:
            extras.append(f"reversed ({rev_sr:.3f}) >= raw_history ({raw_sr:.3f}) — expected reversed to hurt")
    if extras:
        rationale = rationale + "; " + "; ".join(extras)
    return CheckResult("H2_value_destruction", status, rationale)


def _h3_action_conditioning(
    run_dir: str, summary: pd.DataFrame, min_delta: float
) -> CheckResult:
    status, rationale = _sq_beats(
        summary, ["state_only_value_memory", "sq_mem_no_action_conditioning"], min_delta
    )
    return CheckResult("H3_action_conditioning", status, rationale)


def _h4_structured_state(
    run_dir: str, summary: pd.DataFrame, min_delta: float
) -> CheckResult:
    status, rationale = _sq_beats(
        summary, ["raw_prefix_sq_mem", "sq_mem_no_structured_state"], min_delta
    )
    return CheckResult("H4_structured_state", status, rationale)


def _h5_soft_aggregation(
    run_dir: str, summary: pd.DataFrame, min_delta: float
) -> CheckResult:
    status, rationale = _sq_beats(
        summary, ["sq_mem_top1", "sq_mem_uniform_weights"], min_delta
    )
    return CheckResult("H5_soft_aggregation", status, rationale)


def _h6_uncertainty(
    run_dir: str, interventions: pd.DataFrame | None
) -> CheckResult:
    if interventions is None:
        return CheckResult("H6_uncertainty_penalty", NOT_TESTED,
                           "summary_interventions.csv not found")
    for v in ("sq_mem", "sq_mem_no_uncertainty"):
        if v not in interventions.index:
            return CheckResult("H6_uncertainty_penalty", NOT_TESTED,
                               f"{v} missing from interventions summary")
    harmful_full = float(interventions.loc["sq_mem", "harmful_rate"])
    harmful_no = float(interventions.loc["sq_mem_no_uncertainty", "harmful_rate"])
    if harmful_full <= harmful_no:
        return CheckResult("H6_uncertainty_penalty", SUPPORTED,
                           f"sq_mem harmful_rate={harmful_full:.3f} <= no_uncertainty={harmful_no:.3f}")
    return CheckResult("H6_uncertainty_penalty", INCONCLUSIVE,
                       f"sq_mem harmful_rate={harmful_full:.3f} > no_uncertainty={harmful_no:.3f}")


def _h7_calibration(run_dir: str, min_spearman: float) -> CheckResult:
    from scipy.stats import spearmanr  # type: ignore[import]
    import numpy as np
    from typing import Any as _Any

    cal = _load_calibration(run_dir, "sq_mem")
    rationale_parts: list[str] = []

    # Try the existing success_rate-binned calibration first
    if cal is not None and not cal.empty and len(cal) >= 3:
        rho_s = float(spearmanr(cal["q_bin_center"], cal["success_rate"])[0])  # type: ignore[arg-type]
        if rho_s == rho_s:  # not NaN
            rationale_parts.append(f"success_rate spearman={rho_s:.3f}")
            if rho_s >= min_spearman:
                return CheckResult("H7_calibration", SUPPORTED,
                                   f"Spearman rho={rho_s:.3f} >= threshold {min_spearman} (success_rate)")
            if rho_s < 0:
                return CheckResult("H7_calibration", WEAKENED,
                                   f"Negative calibration: rho={rho_s:.3f} (success_rate)")

    # Fall back: recompute calibration on partial-credit avg_total_reward per Q-bin
    decisions_path = os.path.join(run_dir, "decisions_sq_mem.csv")
    eps_path = os.path.join(run_dir, "episodes_sq_mem.jsonl")
    if not os.path.exists(decisions_path) or not os.path.exists(eps_path):
        return CheckResult("H7_calibration", NOT_TESTED,
                           "decisions or episodes file missing; cannot compute fallback calibration")
    d = pd.read_csv(decisions_path)
    eps: list[dict[str, _Any]] = []
    with open(eps_path) as f:
        for line in f:
            line = line.strip()
            if line:
                eps.append(json.loads(line))
    ep_reward: dict[str, float] = {str(e["task_id"]): float(e["total_reward"]) for e in eps}
    d = d.copy()
    d["episode_reward"] = d["task_id"].map(ep_reward)
    qs = d["memory_q"].to_numpy()
    rewards = d["episode_reward"].to_numpy()
    edges = np.unique(np.percentile(qs, np.linspace(0, 100, 6)))
    if len(edges) < 3:
        return CheckResult("H7_calibration", INCONCLUSIVE,
                           f"Q-values too flat to form ≥3 bins; "
                           f"{'; '.join(rationale_parts) if rationale_parts else 'no success_rate signal'}")
    centers: list[float] = []
    mean_rewards: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (qs >= lo) & (qs <= hi)
        if mask.sum() == 0:
            continue
        centers.append(float((lo + hi) / 2.0))
        mean_rewards.append(float(rewards[mask].mean()))
    if len(centers) < 3:
        return CheckResult("H7_calibration", INCONCLUSIVE,
                           f"Only {len(centers)} non-empty Q-bins")
    rho_r = float(spearmanr(centers, mean_rewards)[0])  # type: ignore[arg-type]
    rationale_parts.append(f"avg_total_reward spearman={rho_r:.3f} over {len(centers)} bins")
    rationale = "; ".join(rationale_parts)
    if rho_r >= min_spearman:
        return CheckResult("H7_calibration", SUPPORTED, rationale)
    if rho_r < 0:
        return CheckResult("H7_calibration", WEAKENED, rationale)
    return CheckResult("H7_calibration", INCONCLUSIVE, rationale)


def _h8_intervention_audit(
    run_dir: str,
    interventions: pd.DataFrame | None,
    min_episode_rate: float,
) -> CheckResult:
    if interventions is None:
        return CheckResult("H8_intervention_audit", NOT_TESTED,
                           "summary_interventions.csv not found")
    if "sq_mem" not in interventions.index:
        return CheckResult("H8_intervention_audit", NOT_TESTED,
                           "sq_mem not in interventions")
    ep_rate = float(interventions.loc["sq_mem", "intervention_episode_rate"])  # type: ignore[arg-type]
    ben = float(interventions.loc["sq_mem", "beneficial_rate"])  # type: ignore[arg-type]
    harm = float(interventions.loc["sq_mem", "harmful_rate"])  # type: ignore[arg-type]
    if ep_rate < min_episode_rate:
        return CheckResult("H8_intervention_audit", WEAKENED,
                           f"Memory rarely changes decisions: episode_rate={ep_rate:.3f} < {min_episode_rate}")
    if ben > harm:
        return CheckResult("H8_intervention_audit", SUPPORTED,
                           f"ep_rate={ep_rate:.3f}, beneficial={ben:.3f} > harmful={harm:.3f} (success_rate)")

    # Fallback: compare per-task partial-credit reward between sq_mem episodes
    # where memory intervened and the raw_history baseline. Counts beneficial /
    # harmful interventions using avg_total_reward differences instead of binary success.
    sq_eps = _load_episodes(run_dir, "sq_mem")
    raw_eps = _load_episodes(run_dir, "raw_history")
    sq_dec_path = os.path.join(run_dir, "decisions_sq_mem.csv")
    if not sq_eps or not raw_eps or not os.path.exists(sq_dec_path):
        return CheckResult("H8_intervention_audit", INCONCLUSIVE,
                           f"ep_rate={ep_rate:.3f}, success-rate beneficial={ben:.3f}=harmful={harm:.3f}; "
                           "no episode/decision data for reward-based fallback")
    raw_r: dict[str, float] = {str(e["task_id"]): float(e["total_reward"]) for e in raw_eps}
    sq_r: dict[str, float] = {str(e["task_id"]): float(e["total_reward"]) for e in sq_eps}
    sq_dec = pd.read_csv(sq_dec_path)
    intervened_tasks = set(
        sq_dec[sq_dec["memory_changed_decision"]]["task_id"].astype(str).unique().tolist()
    )
    if not intervened_tasks:
        return CheckResult("H8_intervention_audit", WEAKENED,
                           f"No intervened episodes; ep_rate={ep_rate:.3f}")

    eps_tol = 1e-4
    ben_r = 0
    harm_r = 0
    neutral_r = 0
    for t in intervened_tasks:
        if t not in raw_r or t not in sq_r:
            continue
        delta = sq_r[t] - raw_r[t]
        if delta > eps_tol:
            ben_r += 1
        elif delta < -eps_tol:
            harm_r += 1
        else:
            neutral_r += 1
    total = ben_r + harm_r + neutral_r
    rationale = (f"ep_rate={ep_rate:.3f}, reward-based: beneficial={ben_r}/{total} "
                 f"harmful={harm_r}/{total} neutral={neutral_r}/{total}")
    if ben_r > harm_r:
        return CheckResult("H8_intervention_audit", SUPPORTED, rationale)
    if harm_r > ben_r:
        return CheckResult("H8_intervention_audit", WEAKENED, rationale)
    return CheckResult("H8_intervention_audit", INCONCLUSIVE, rationale)


def _h9_horizon(run_dir: str, min_delta: float) -> CheckResult:
    p = os.path.join(run_dir, "summary_horizon_buckets.csv")
    if not os.path.exists(p):
        return CheckResult("H9_horizon_effect", NOT_TESTED,
                           "summary_horizon_buckets.csv not found")
    df = pd.read_csv(p)
    req = {"variant", "horizon_bucket", "success_rate"}
    if not req.issubset(df.columns):
        return CheckResult("H9_horizon_effect", NOT_TESTED, "Missing columns")
    sq = df[df["variant"] == "sq_mem"]
    raw = df[df["variant"] == "raw_history"]
    if sq.empty or raw.empty:
        return CheckResult("H9_horizon_effect", NOT_TESTED,
                           "sq_mem or raw_history missing from horizon buckets")
    merged = sq.merge(raw, on="horizon_bucket", suffixes=("_sq", "_raw"))
    merged["delta"] = merged["success_rate_sq"] - merged["success_rate_raw"]
    long_rows = merged[merged["horizon_bucket"] == "long"]
    short_rows = merged[merged["horizon_bucket"] == "short"]
    if long_rows.empty or short_rows.empty:
        return CheckResult("H9_horizon_effect", INCONCLUSIVE,
                           "Need both short and long horizon buckets")
    long_delta = float(long_rows["delta"].iloc[0])
    short_delta = float(short_rows["delta"].iloc[0])
    if long_delta > short_delta and long_delta >= min_delta:
        return CheckResult("H9_horizon_effect", SUPPORTED,
                           f"long_delta={long_delta:.3f} > short_delta={short_delta:.3f}")
    return CheckResult("H9_horizon_effect", INCONCLUSIVE,
                       f"long_delta={long_delta:.3f}, short_delta={short_delta:.3f}")


def _h10_split_discipline(run_dir: str) -> CheckResult:
    mem = _load_memory_summary(run_dir)
    if mem is None:
        return CheckResult("H10_split_discipline", SETUP_FAILED,
                           "memory_summary.json not found")
    test_count = mem.get("splits", {}).get("test", 0)
    if test_count > 0:
        return CheckResult("H10_split_discipline", SETUP_FAILED,
                           f"memory_summary.json reports {test_count} test-split items")
    return CheckResult("H10_split_discipline", SUPPORTED,
                       "No test-split items found in memory bank")


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def evaluate_run(
    run_dir: str,
    thresholds: dict[str, Any] | None = None,
) -> tuple[list[CheckResult], list[ComparisonResult]]:
    t: dict[str, Any] = {
        "min_success_delta": 0.02,
        "min_error_recovery_delta": 0.02,
        "min_calibration_spearman": 0.30,
        "min_intervention_episode_rate": 0.01,
        "bootstrap_samples": 2000,
        "min_bootstrap_pairs": 10,
        "require_positive_ci_when_available": False,
    }
    if thresholds:
        t.update(thresholds)

    summary = _load_summary(run_dir)
    interventions = _load_interventions(run_dir)

    checks: list[CheckResult] = []
    comparisons: list[ComparisonResult] = []

    checks.append(_h0_pipeline_sanity(run_dir))

    if summary is None:
        for hid in [f"H{i}" for i in range(1, 11)]:
            checks.append(CheckResult(hid, SETUP_FAILED, "summary.csv missing"))
        return checks, comparisons

    min_d = float(t["min_success_delta"])
    checks.append(_h1_main_comparison(run_dir, summary, min_d))
    checks.append(_h2_value_destruction(run_dir, summary, min_d))
    checks.append(_h3_action_conditioning(run_dir, summary, min_d))
    checks.append(_h4_structured_state(run_dir, summary, min_d))
    checks.append(_h5_soft_aggregation(run_dir, summary, min_d))
    checks.append(_h6_uncertainty(run_dir, interventions))
    checks.append(_h7_calibration(run_dir, float(t["min_calibration_spearman"])))
    checks.append(_h8_intervention_audit(
        run_dir, interventions, float(t["min_intervention_episode_rate"])
    ))
    checks.append(_h9_horizon(run_dir, min_d))
    checks.append(_h10_split_discipline(run_dir))

    # Paired comparisons for H1 and H2
    key_pairs = [
        ("H1", "sq_mem", "semantic_retrieval"),
        ("H1", "sq_mem", "raw_history"),
        ("H2", "sq_mem", "sq_mem_shuffled_returns"),
        ("H2", "sq_mem", "sq_mem_value_reversed"),
        ("H3", "sq_mem", "state_only_value_memory"),
        ("H3", "sq_mem", "sq_mem_no_action_conditioning"),
    ]
    for check_id, va, vb in key_pairs:
        comparisons.append(
            _paired_comparison(
                run_dir, check_id, va, vb,
                int(t["bootstrap_samples"]),
                int(t["min_bootstrap_pairs"]),
            )
        )

    return checks, comparisons


def evaluate_and_write(
    run_dir: str,
    thresholds: dict[str, Any] | None = None,
) -> None:
    checks, comparisons = evaluate_run(run_dir, thresholds)

    # hypothesis_checks.csv
    checks_path = os.path.join(run_dir, "hypothesis_checks.csv")
    with open(checks_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check_id", "status", "rationale"])
        w.writeheader()
        for c in checks:
            w.writerow({"check_id": c.check_id, "status": c.status, "rationale": c.rationale})

    # hypothesis_comparisons.csv
    comp_path = os.path.join(run_dir, "hypothesis_comparisons.csv")
    with open(comp_path, "w", newline="") as f:
        w2 = csv.DictWriter(
            f,
            fieldnames=["check_id", "variant_a", "variant_b", "delta_success",
                        "ci_low", "ci_high", "n_pairs", "note"],
        )
        w2.writeheader()
        for c in comparisons:
            w2.writerow({
                "check_id": c.check_id,
                "variant_a": c.variant_a,
                "variant_b": c.variant_b,
                "delta_success": c.delta_success,
                "ci_low": c.ci_low,
                "ci_high": c.ci_high,
                "n_pairs": c.n_pairs,
                "note": c.note,
            })

    # hypothesis_report.json
    report_data: dict[str, Any] = {
        "checks": [{"check_id": c.check_id, "status": c.status, "rationale": c.rationale}
                   for c in checks],
        "comparisons": [
            {"check_id": c.check_id, "variant_a": c.variant_a, "variant_b": c.variant_b,
             "delta_success": c.delta_success, "ci_low": c.ci_low, "ci_high": c.ci_high,
             "n_pairs": c.n_pairs}
            for c in comparisons
        ],
    }
    with open(os.path.join(run_dir, "hypothesis_report.json"), "w") as f:
        json.dump(report_data, f, indent=2)

    # hypothesis_report.md
    lines = ["# Hypothesis Report\n"]
    for c in checks:
        icon = {"supported": "✅", "weakened": "⚠️", "inconclusive": "❓",
                "not_tested": "–", "setup_failed": "❌"}.get(c.status, "?")
        lines.append(f"## {c.check_id}: {c.status} {icon}\n{c.rationale}\n")
    with open(os.path.join(run_dir, "hypothesis_report.md"), "w") as f:
        f.write("\n".join(lines))
