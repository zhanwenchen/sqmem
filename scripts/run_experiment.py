#!/usr/bin/env python3
"""Main entry point for running an SQ-Mem experiment."""
import argparse
import sys

sys.path.insert(0, ".")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an SQ-Mem experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output-dir", default="results", help="Parent output directory")
    parser.add_argument(
        "--run-id-prefix",
        default="",
        help=(
            "String prepended to the run_id when constructing the output "
            "directory. Use a batch timestamp (e.g. '20260520235909_') to "
            "group multiple runs from one overnight queue under a shared "
            "prefix."
        ),
    )
    args = parser.parse_args()

    from sq_mem_experiments.evaluation.runner import run_from_config

    run_from_config(args.config, args.output_dir, run_id_prefix=args.run_id_prefix)


if __name__ == "__main__":
    main()
