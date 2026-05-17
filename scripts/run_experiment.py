#!/usr/bin/env python3
"""Main entry point for running an SQ-Mem experiment."""
import argparse
import sys

sys.path.insert(0, ".")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an SQ-Mem experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output-dir", default="results", help="Parent output directory")
    args = parser.parse_args()

    from sq_mem_experiments.evaluation.runner import run_from_config

    run_from_config(args.config, args.output_dir)


if __name__ == "__main__":
    main()
