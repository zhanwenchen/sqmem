# Claim Ledger

This document maps each paper claim to the repo artifact that provides evidence for or against it.

| Paper claim | How it can be supported | How it can be weakened | Primary artifact |
|---|---|---|---|
| Memory functions as a soft Q table | `sq_mem` > `semantic_retrieval` on success rate | `semantic_retrieval` matches `sq_mem` | `summary.csv` |
| Downstream returns are the value signal | `sq_mem` > shuffled / reversed / zero / no-return controls | Destroying returns has no effect | `summary_value_destruction.csv` |
| Action conditioning matters (Q(s,a) vs V(s)) | `sq_mem` > `state_only_value_memory` and `sq_mem_no_action_conditioning` | State-only retrieval matches full SQ-Mem | `decisions_sq_mem.csv`, `calibration_sq_mem.csv` |
| Structured prefix state matters | `sq_mem` > `raw_prefix_sq_mem` and `sq_mem_no_structured_state` | Raw-transcript retrieval performs equally | `summary.csv` |
| Retrieval relevance matters | `sq_mem` > `sq_mem_random_memory` | Random memory performs similarly | `summary_value_destruction.csv` |
| Soft aggregation beats top-1 | `sq_mem` has better calibration than `sq_mem_top1` | Top-1 is equally calibrated | `calibration_sq_mem.csv`, `calibration_sq_mem_top1.csv` |
| Memory actually changes decisions | Intervention episode rate > threshold; beneficial > harmful | Memory rarely changes the base action | `summary_interventions.csv` |
| Memory values are calibrated | Monotone Spearman correlation in calibration bins | Flat or inverted calibration curve | `calibration_sq_mem.csv` |
| Gains are larger on long-horizon tasks | Long-bucket delta > short-bucket delta | Gains only on short / easy tasks | `summary_horizon_buckets.csv` |
| No test leakage | memory_summary.json shows only train-split items | Test task IDs appear in memory bank | `memory_summary.json`, `memory_bank.jsonl` |

## Automated evaluation

All claims are checked automatically after every run by `hypothesis_testing.py`.
Each claim maps to a check ID (H0–H10) in `hypothesis_checks.csv`.

A claim is paper-ready only when the automated report **and** a qualitative audit of
`decisions_sq_mem.csv` agree.

## Minimum required result pattern

For the Soft-Q Memory interpretation to be credible:

- `H1_main_sq_mem_comparison`: supported
- `H2_value_destruction`: supported
- `H7_calibration`: supported
- `H8_intervention_audit`: supported
- `H10_split_discipline`: supported
