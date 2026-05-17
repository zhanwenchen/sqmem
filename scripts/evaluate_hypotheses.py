#!/usr/bin/env python3
"""Re-run automated hypothesis checks on a completed result directory."""
import argparse
import sys

sys.path.insert(0, ".")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate hypothesis ledger on a completed run directory"
    )
    parser.add_argument("run_dir", help="Path to a completed run directory")
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a threshold, e.g. --threshold min_success_delta=0.05",
    )
    args = parser.parse_args()

    thresholds: dict[str, object] = {}
    for kv in args.threshold:
        key, _, raw = kv.partition("=")
        # Try to coerce to float or bool
        if raw.lower() in ("true", "false"):
            thresholds[key] = raw.lower() == "true"
        else:
            try:
                thresholds[key] = float(raw)
            except ValueError:
                thresholds[key] = raw

    from sq_mem_experiments.evaluation.hypothesis_testing import evaluate_and_write

    evaluate_and_write(args.run_dir, thresholds or None)
    print(f"Hypothesis report written to: {args.run_dir}/hypothesis_report.md")


if __name__ == "__main__":
    main()
