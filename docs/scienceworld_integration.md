# ScienceWorld Integration

## Requirements

```bash
pip install "sq_mem_experiments[scienceworld]"
# Java >= 8 must be on PATH
java -version
```

## Task splits

Train and test variation IDs must be disjoint. The smoke config uses:

- train: `[0, 1, 2, 3, 4]`
- test: `[20, 21, 22, 23, 24]`

The runner enforces this with an assertion before writing any test results.

## Reward mode

`score_delta` (default): each step reward is `(score_t - score_{t-1}) / 100`.
The return-to-go stored in each memory item therefore represents future score
improvement starting from that prefix-action pair.

## Memory builder

`ScienceWorldRandomMemoryBuilderAgent` mixes a keyword-heuristic policy
(`heuristic_prob`) with random exploration. For paper-quality evidence, replace
it with expert trajectories, official baseline trajectories, or a frozen LLM
policy. SQ-Mem needs contrastive memory — both successes and failures — so the
memory builder should not be pure-random (all zero returns) or pure-expert
(no negative signals).

## State compiler

`compile_scienceworld()` in `memory/compiler.py` builds the structured prefix
state from: task goal, current ScienceWorld score, step count, current
observation, and last 3 actions. This is the state representation used for
retrieval in the `sq_mem` variant. The `raw_prefix_sq_mem` variant bypasses
this and uses the raw interleaved observation/action transcript instead.

## Extending to new task types

1. Add the task name to `task_names` in the config YAML.
2. Verify that `ScienceWorldAdapter.make_tasks()` generates non-overlapping
   train/test IDs for those tasks.
3. Optionally tune `_ACTION_KEYWORDS` in `agents/scienceworld_agents.py` to
   improve the base policy heuristic for that task type.

## Known limitations of the scaffold

- The base policy is a keyword heuristic with no LLM. Success rates will be
  low without a strong policy. The main purpose of the scaffold is to verify
  that the **relative ordering** of variants matches the theoretical predictions,
  not to achieve high absolute success rates.
- ScienceWorld requires Java and can be slow. Use `max_steps: 30` and a small
  number of test variations for smoke runs.
- The `ScienceWorldRandomMemoryBuilderAgent` collects low-quality trajectories.
  Replace with a stronger policy before claiming paper-level evidence.
