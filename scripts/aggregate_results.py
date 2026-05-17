#!/usr/bin/env python3
"""Aggregate summaries and hypothesis checks across multiple run directories."""
import argparse
import csv
import os
import sys
from typing import Any

sys.path.insert(0, ".")


def _load_csv(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate results across runs")
    parser.add_argument("run_dirs", nargs="+", help="Run directories to aggregate")
    parser.add_argument("--out", default="results/aggregate_summary.csv")
    parser.add_argument(
        "--hypotheses-out", default="results/aggregate_hypothesis_checks.csv"
    )
    args = parser.parse_args()

    summary_rows: list[dict[str, Any]] = []
    hypothesis_rows: list[dict[str, Any]] = []

    for run_dir in args.run_dirs:
        run_id = os.path.basename(run_dir.rstrip("/"))
        for row in _load_csv(os.path.join(run_dir, "summary.csv")):
            row["run_id"] = run_id
            summary_rows.append(row)
        for row in _load_csv(os.path.join(run_dir, "hypothesis_checks.csv")):
            row["run_id"] = run_id
            hypothesis_rows.append(row)

    def _write(path: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            print(f"  (no data for {path})")
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {len(rows)} rows → {path}")

    _write(args.out, summary_rows)
    _write(args.hypotheses_out, hypothesis_rows)


if __name__ == "__main__":
    main()
