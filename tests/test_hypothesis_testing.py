"""Tests for the automated hypothesis evaluator using synthetic artifacts."""
import csv
import json
import os
import pathlib

import pytest

from sq_mem_experiments.evaluation.hypothesis_testing import (
    INCONCLUSIVE,
    NOT_TESTED,
    SETUP_FAILED,
    SUPPORTED,
    WEAKENED,
    evaluate_and_write,
    evaluate_run,
)


# ---------------------------------------------------------------------------
# Helpers to create synthetic run directories
# ---------------------------------------------------------------------------

def _write_summary(run_dir: str, rows: list[dict[str, object]]) -> None:
    path = os.path.join(run_dir, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_memory_summary(run_dir: str, data: dict[str, object]) -> None:
    with open(os.path.join(run_dir, "memory_summary.json"), "w") as f:
        json.dump(data, f)


def _write_memory_bank(run_dir: str, items: list[dict[str, object]]) -> None:
    with open(os.path.join(run_dir, "memory_bank.jsonl"), "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def _write_interventions(run_dir: str, rows: list[dict[str, object]]) -> None:
    path = os.path.join(run_dir, "summary_interventions.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_calibration(run_dir: str, variant: str, rows: list[dict[str, object]]) -> None:
    path = os.path.join(run_dir, f"calibration_{variant}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _base_run_dir(tmp_path: pathlib.Path, subname: str = "run") -> str:
    d = str(tmp_path / subname)
    os.makedirs(d)
    return d


# ---------------------------------------------------------------------------
# H0: pipeline sanity
# ---------------------------------------------------------------------------

def test_h0_fails_without_summary(tmp_path: pathlib.Path) -> None:
    run_dir = _base_run_dir(tmp_path, "h0_fail")
    checks, _ = evaluate_run(run_dir)
    h0 = next(c for c in checks if c.check_id == "H0_pipeline_sanity")
    assert h0.status == SETUP_FAILED


def test_h0_passes_with_all_artifacts(tmp_path: pathlib.Path) -> None:
    run_dir = _base_run_dir(tmp_path, "h0_pass")
    _write_summary(run_dir, [{"variant": "sq_mem", "success_rate": 0.5}])
    _write_memory_summary(run_dir, {"splits": {"train": 10}, "total_items": 10, "task_ids": []})
    _write_memory_bank(run_dir, [])
    checks, _ = evaluate_run(run_dir)
    h0 = next(c for c in checks if c.check_id == "H0_pipeline_sanity")
    assert h0.status == SUPPORTED


def test_h0_fails_on_test_leakage(tmp_path: pathlib.Path) -> None:
    run_dir = _base_run_dir(tmp_path, "h0_leak")
    _write_summary(run_dir, [{"variant": "sq_mem", "success_rate": 0.5}])
    _write_memory_summary(run_dir, {"splits": {"train": 5, "test": 2}, "total_items": 7, "task_ids": []})
    _write_memory_bank(run_dir, [])
    checks, _ = evaluate_run(run_dir)
    h0 = next(c for c in checks if c.check_id == "H0_pipeline_sanity")
    assert h0.status == SETUP_FAILED


# ---------------------------------------------------------------------------
# H1: main SQ-Mem comparison
# ---------------------------------------------------------------------------

def test_h1_supported_when_sq_mem_wins(tmp_path: pathlib.Path) -> None:
    run_dir = _base_run_dir(tmp_path, "h1_sup")
    _write_summary(run_dir, [
        {"variant": "sq_mem",             "success_rate": 0.6},
        {"variant": "semantic_retrieval", "success_rate": 0.4},
        {"variant": "summary_memory",     "success_rate": 0.35},
        {"variant": "raw_history",        "success_rate": 0.3},
    ])
    _write_memory_summary(run_dir, {"splits": {"train": 10}, "total_items": 10, "task_ids": []})
    _write_memory_bank(run_dir, [])
    checks, _ = evaluate_run(run_dir)
    h1 = next(c for c in checks if c.check_id == "H1_main_sq_mem_comparison")
    assert h1.status == SUPPORTED


def test_h1_not_tested_without_sq_mem(tmp_path: pathlib.Path) -> None:
    run_dir = _base_run_dir(tmp_path, "h1_nt")
    _write_summary(run_dir, [
        {"variant": "raw_history", "success_rate": 0.3},
    ])
    _write_memory_summary(run_dir, {"splits": {"train": 5}, "total_items": 5, "task_ids": []})
    _write_memory_bank(run_dir, [])
    checks, _ = evaluate_run(run_dir)
    h1 = next(c for c in checks if c.check_id == "H1_main_sq_mem_comparison")
    assert h1.status == NOT_TESTED


def test_h1_weakened_when_semantic_matches(tmp_path: pathlib.Path) -> None:
    run_dir = _base_run_dir(tmp_path, "h1_weak")
    _write_summary(run_dir, [
        {"variant": "sq_mem",             "success_rate": 0.50},
        {"variant": "semantic_retrieval", "success_rate": 0.50},
        {"variant": "summary_memory",     "success_rate": 0.50},
        {"variant": "raw_history",        "success_rate": 0.50},
    ])
    _write_memory_summary(run_dir, {"splits": {"train": 5}, "total_items": 5, "task_ids": []})
    _write_memory_bank(run_dir, [])
    checks, _ = evaluate_run(run_dir, {"min_success_delta": 0.02})
    h1 = next(c for c in checks if c.check_id == "H1_main_sq_mem_comparison")
    # Delta is 0, below threshold → INCONCLUSIVE or WEAKENED
    assert h1.status in (INCONCLUSIVE, WEAKENED)


# ---------------------------------------------------------------------------
# H2: value destruction
# ---------------------------------------------------------------------------

def test_h2_supported_when_destruction_hurts(tmp_path: pathlib.Path) -> None:
    run_dir = _base_run_dir(tmp_path, "h2_sup")
    _write_summary(run_dir, [
        {"variant": "sq_mem",                  "success_rate": 0.7},
        {"variant": "sq_mem_shuffled_returns", "success_rate": 0.3},
        {"variant": "sq_mem_value_reversed",   "success_rate": 0.1},
        {"variant": "sq_mem_zero_returns",     "success_rate": 0.3},
        {"variant": "sq_mem_no_returns",       "success_rate": 0.3},
        {"variant": "sq_mem_random_memory",    "success_rate": 0.3},
        {"variant": "raw_history",             "success_rate": 0.4},
    ])
    _write_memory_summary(run_dir, {"splits": {"train": 5}, "total_items": 5, "task_ids": []})
    _write_memory_bank(run_dir, [])
    checks, _ = evaluate_run(run_dir)
    h2 = next(c for c in checks if c.check_id == "H2_value_destruction")
    assert h2.status == SUPPORTED


# ---------------------------------------------------------------------------
# H7: calibration
# ---------------------------------------------------------------------------

def test_h7_supported_with_monotonic_calibration(tmp_path: pathlib.Path) -> None:
    run_dir = _base_run_dir(tmp_path, "h7_sup")
    _write_summary(run_dir, [{"variant": "sq_mem", "success_rate": 0.5}])
    _write_memory_summary(run_dir, {"splits": {"train": 5}, "total_items": 5, "task_ids": []})
    _write_memory_bank(run_dir, [])
    _write_calibration(run_dir, "sq_mem", [
        {"q_bin_low": -1.0, "q_bin_high": -0.3, "q_bin_center": -0.65, "n_decisions": 10, "success_rate": 0.1},
        {"q_bin_low": -0.3, "q_bin_high":  0.3, "q_bin_center":  0.0,  "n_decisions": 20, "success_rate": 0.5},
        {"q_bin_low":  0.3, "q_bin_high":  1.0, "q_bin_center":  0.65, "n_decisions": 15, "success_rate": 0.9},
    ])
    checks, _ = evaluate_run(run_dir, {"min_calibration_spearman": 0.5})
    h7 = next(c for c in checks if c.check_id == "H7_calibration")
    assert h7.status == SUPPORTED


def test_h7_weakened_with_inverted_calibration(tmp_path: pathlib.Path) -> None:
    run_dir = _base_run_dir(tmp_path, "h7_weak")
    _write_summary(run_dir, [{"variant": "sq_mem", "success_rate": 0.5}])
    _write_memory_summary(run_dir, {"splits": {"train": 5}, "total_items": 5, "task_ids": []})
    _write_memory_bank(run_dir, [])
    _write_calibration(run_dir, "sq_mem", [
        {"q_bin_low": -1.0, "q_bin_high": -0.3, "q_bin_center": -0.65, "n_decisions": 10, "success_rate": 0.9},
        {"q_bin_low": -0.3, "q_bin_high":  0.3, "q_bin_center":  0.0,  "n_decisions": 20, "success_rate": 0.5},
        {"q_bin_low":  0.3, "q_bin_high":  1.0, "q_bin_center":  0.65, "n_decisions": 15, "success_rate": 0.1},
    ])
    checks, _ = evaluate_run(run_dir, {"min_calibration_spearman": 0.5})
    h7 = next(c for c in checks if c.check_id == "H7_calibration")
    assert h7.status == WEAKENED


# ---------------------------------------------------------------------------
# H10: split discipline
# ---------------------------------------------------------------------------

def test_h10_setup_failed_on_test_leakage(tmp_path: pathlib.Path) -> None:
    run_dir = _base_run_dir(tmp_path, "h10_fail")
    _write_summary(run_dir, [{"variant": "sq_mem", "success_rate": 0.5}])
    _write_memory_summary(run_dir, {"splits": {"test": 3}, "total_items": 3, "task_ids": []})
    _write_memory_bank(run_dir, [])
    checks, _ = evaluate_run(run_dir)
    h10 = next(c for c in checks if c.check_id == "H10_split_discipline")
    assert h10.status == SETUP_FAILED


# ---------------------------------------------------------------------------
# Output artifacts
# ---------------------------------------------------------------------------

def test_evaluate_and_write_creates_all_files(tmp_path: pathlib.Path) -> None:
    run_dir = _base_run_dir(tmp_path, "write_test")
    _write_summary(run_dir, [{"variant": "sq_mem", "success_rate": 0.5}])
    _write_memory_summary(run_dir, {"splits": {"train": 5}, "total_items": 5, "task_ids": []})
    _write_memory_bank(run_dir, [])

    evaluate_and_write(run_dir)

    for fname in [
        "hypothesis_report.md",
        "hypothesis_report.json",
        "hypothesis_checks.csv",
        "hypothesis_comparisons.csv",
    ]:
        assert os.path.exists(os.path.join(run_dir, fname)), f"Missing: {fname}"


def test_hypothesis_checks_csv_has_all_checks(tmp_path: pathlib.Path) -> None:
    run_dir = _base_run_dir(tmp_path, "checks_csv")
    _write_summary(run_dir, [{"variant": "sq_mem", "success_rate": 0.5}])
    _write_memory_summary(run_dir, {"splits": {"train": 5}, "total_items": 5, "task_ids": []})
    _write_memory_bank(run_dir, [])

    evaluate_and_write(run_dir)

    with open(os.path.join(run_dir, "hypothesis_checks.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    check_ids = {r["check_id"] for r in rows}
    for expected in ["H0_pipeline_sanity", "H1_main_sq_mem_comparison",
                     "H2_value_destruction", "H10_split_discipline"]:
        assert expected in check_ids
