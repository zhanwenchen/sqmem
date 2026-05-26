# Soft-Q Memory Experiments

Runnable scaffold for testing the thesis:

> For online decision-making, the control-relevant part of agent memory functions as a **soft Q table**: it stores evidence about which continuations from similar trajectory prefixes led to better or worse outcomes.

See [paper/soft_q_memory.md](paper/soft_q_memory.md) for the full write-up.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
# For ScienceWorld runs (requires Java >= 8):
pip install -e ".[dev,scienceworld]"
```

Run tests:

```bash
pytest -q
```

---

## Quick start

```bash
# Smoke run (requires scienceworld)
python scripts/run_experiment.py \
  --config configs/scienceworld_smoke.yaml \
  --output-dir results

# Full ablation suite
python scripts/run_experiment.py \
  --config configs/scienceworld_main.yaml \
  --output-dir results

# Re-evaluate hypothesis ledger on existing results
python scripts/evaluate_hypotheses.py results/scienceworld_smoke

# Aggregate across seeds
python scripts/aggregate_results.py \
  results/scienceworld_seed0 results/scienceworld_seed1 \
  --out results/aggregate_summary.csv \
  --hypotheses-out results/aggregate_hypothesis_checks.csv
```

---

## The estimator

$$
\hat{Q}_{\mathcal{M}}(s_t,a)
= \sum_{i \in \mathcal{N}_R(z_t,u)} w_i(z_t,u)\,G_i
$$

At decision time the agent reranks candidate actions:

$$
a_t = \arg\max_{a \in \mathcal{A}_t}
\left[
S_\theta(a \mid \tau_{\leq t},s_t)
+ \lambda\,\hat{Q}_{\mathcal{M}}(s_t,a)
- \rho\,\hat{\sigma}_{\mathcal{M}}(s_t,a)
\right]
$$

No Bellman update, no Q-network, no entropy-regularized policy.

---

## Variants

| Variant | What it tests |
|---|---|
| `raw_history` | Base policy with no memory |
| `summary_memory` | Current-prefix compression, no prior episodes |
| `semantic_retrieval` | Retrieval without using returns |
| `state_only_value_memory` | V(s) instead of Q(s,a) |
| `raw_prefix_sq_mem` | Raw transcript instead of structured state |
| `sq_mem` | **Full SQ-Mem** |
| `sq_mem_no_returns` | Remove value labels |
| `sq_mem_no_action_conditioning` | Remove action matching |
| `sq_mem_no_structured_state` | Remove prefix-to-state compilation |
| `sq_mem_no_uncertainty` | Remove uncertainty penalty |
| `sq_mem_top1` | Nearest-neighbour lookup |
| `sq_mem_uniform_weights` | Uniform top-R averaging |
| `sq_mem_random_memory` | Break retrieval relevance |
| `sq_mem_shuffled_returns` | Break return-row mapping |
| `sq_mem_value_reversed` | Invert the value signal |
| `sq_mem_zero_returns` | Keep retrieval, erase values |

The key falsification test: `sq_mem_shuffled_returns` and `sq_mem_value_reversed`
should both hurt relative to `sq_mem`. If they don't, the soft-Q-table
interpretation is not supported.

---

## Output files

```
results/<run_id>/
├── config_resolved.yaml
├── memory_bank.jsonl          # prefix | action | return rows
├── memory_summary.json
├── summary.csv                # per-variant aggregate metrics
├── summary_interventions.csv  # did memory change decisions?
├── summary_value_destruction.csv
├── summary_horizon_buckets.csv
├── decisions_all.csv
├── decisions_<variant>.csv    # decision-level logs with retrieved returns
├── calibration_<variant>.csv
├── episodes_<variant>.jsonl
├── hypothesis_report.md       # automated claim evaluation
├── hypothesis_report.json
├── hypothesis_checks.csv      # H0–H10 status
└── hypothesis_comparisons.csv # paired bootstrap CIs
```

---

## Repository structure

```
sq_mem_experiments/
├── schema.py                        # shared dataclasses
├── memory/
│   ├── embeddings.py                # deterministic hash embedder
│   ├── compiler.py                  # prefix-to-state compilers
│   ├── store.py                     # JSONL bank + return transforms
│   └── soft_q_memory.py             # core Q estimator
├── envs/
│   ├── base.py
│   ├── scienceworld_adapter.py
│   └── {appworld,webarena,agentgym}_adapter.py  # stubs
├── agents/
│   ├── base.py
│   └── scienceworld_agents.py       # all variants + factory
└── evaluation/
    ├── rollout.py                   # episode rollout + return-to-go
    ├── memory_builder.py            # training→memory conversion
    ├── metrics.py                   # success, calibration, interventions
    ├── hypothesis_testing.py        # automated H0–H10 checks
    └── runner.py                    # 7-stage orchestrator
scripts/
├── run_experiment.py
├── evaluate_hypotheses.py
└── aggregate_results.py
configs/
├── scienceworld_smoke.yaml
└── scienceworld_main.yaml
tests/
├── test_core_components.py          # estimator + return-destruction
├── test_memory_mechanism.py         # action conditioning + split checks
└── test_hypothesis_testing.py       # automated evaluator on synthetic data
```

---

## What would falsify this?

- `semantic_retrieval` ≈ `sq_mem`  (retrieval alone explains the gains)
- `sq_mem_shuffled_returns` ≈ `sq_mem`  (returns don't matter)
- `sq_mem_value_reversed` does not hurt  (wrong values don't mislead)
- Calibration is flat or inverted  (Q̂ doesn't predict outcomes)
- Memory rarely changes the base action  (memory has no effect)

See [docs/claim_ledger.md](docs/claim_ledger.md) for the full ledger.


#### Notes

```bash
hf download unsloth/Qwen3.5-4B-GGUF --local-dir qwen3.5_4b Qwen3.5-4B-Q4_K_M.gguf
HF_HUB_ENABLE_HF_TRANSFER=1 hf download unsloth/Qwen3.5-9B-GGUF --local-dir qwen3.5_9b Qwen3.5-9B-UD-Q4_K_XL.gguf

```

```bash
/Users/zhanwenchen/llama.cpp/build/bin/llama-server unsloth/Qwen2.5-3B-Instruct-GGUF:Q4_K_M --port 8080 --ctx-size 2048 --api-key local
```

```bash
/Users/zhanwenchen/llama.cpp/build/bin/llama-server qwen3.5_4b/Qwen3.5-4B-Q4_K_M.gguf --port 8080 --ctx-size 2048 --api-key local
```

```bash
(sqmem) (base) ➜  sqmem git:(main) ✗ /Users/zhanwenchen/llama.cpp/build/bin/llama-server \
  -m qwen3.5_4b/Qwen3.5-4B-Q4_K_M.gguf \
  --port 8080 --ctx-size 2048 --api-key local
0.00.078.739 I log_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
0.00.078.741 I device_info:
0.00.078.747 I   - MTL0    : Apple M2 Max (79626 MiB, 79625 MiB free)
0.00.078.747 I   - BLAS    : Accelerate (0 MiB, 0 MiB free)
0.00.078.753 I   - CPU     : Apple M2 Max (98304 MiB, 98304 MiB free)
0.00.078.913 I system_info: n_threads = 8 (n_threads_batch = 8) / 12 | MTL : EMBED_LIBRARY = 1 | CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1 | DOTPROD = 1 | ACCELERATE = 1 | REPACK = 1 |
0.00.078.915 I srv          main: n_parallel is set to auto, using n_parallel = 4 and kv_unified = true
0.00.078.985 I srv          init: running without SSL
0.00.080.006 I srv          init: api_keys: ****ocal
0.00.080.020 I srv          init: using 11 threads for HTTP server
0.00.080.650 I srv         start: binding port with default address family
0.00.082.303 I srv          main: loading model
0.00.082.517 I srv    load_model: loading model 'qwen3.5_4b/Qwen3.5-4B-Q4_K_M.gguf'
0.00.083.238 I common_init_result: fitting params to device memory ...
0.00.083.240 I common_init_result: (for bugs during this step try to reproduce them with -fit off, or provide --verbose logs if the bug only occurs with -fit on)
0.00.839.416 W llama_context: n_ctx_seq (2048) < n_ctx_train (262144) -- the full capacity of the model will not be utilized
0.00.889.790 W common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
0.01.224.509 I srv    load_model: initializing slots, n_slots = 4
0.01.250.717 W srv    load_model: speculative decoding will use checkpoints
0.01.250.741 W common_speculative_init: no implementations specified for speculative decoding
0.01.250.742 I slot   load_model: id  0 | task -1 | new slot, n_ctx = 2048
0.01.250.775 I slot   load_model: id  1 | task -1 | new slot, n_ctx = 2048
0.01.250.777 I slot   load_model: id  2 | task -1 | new slot, n_ctx = 2048
0.01.250.777 I slot   load_model: id  3 | task -1 | new slot, n_ctx = 2048
0.01.250.812 I srv    load_model: prompt cache is enabled, size limit: 8192 MiB
0.01.250.813 I srv    load_model: use `--cache-ram 0` to disable the prompt cache
0.01.250.814 I srv    load_model: for more info see https://github.com/ggml-org/llama.cpp/pull/16391
0.01.250.843 I srv          init: idle slots will be saved to prompt cache and cleared upon starting a new task
0.01.269.846 I init: chat template, example_format: '<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
<think>
'
0.01.278.563 I srv          init: init: chat template, thinking = 1
0.01.278.577 I srv          main: model loaded
0.01.278.581 I srv          main: server is listening on http://127.0.0.1:8080
0.01.278.781 I srv  update_slots: all slots are idle
^C7.27.073.438 I srv    operator(): operator(): cleaning up before exit...
(sqmem) (base) ➜  sqmem git:(main) ✗
```


```bash
uv pip install openai sentence-transformers
uv pip install alfworld
```


First make sure the service is running:

```bash
/Users/zhanwenchen/llama.cpp/build/bin/llama-server -m /Users/zhanwenchen/sqmem/qwen3.5_4b/Qwen3.5-4B-Q4_K_M.gguf --port 8080 --ctx-size 2048 --api-key local
/Users/zhanwenchen/llama.cpp/build/bin/llama-server \
  -m /Users/zhanwenchen/sqmem/qwen3.5_9b/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  --port 8080 \
  --ctx-size 8192 \
  -ngl 999 \
  --api-key local \
  --chat-template-kwargs '{"enable_thinking":false}'

```

Then run the code:

```bash
rm -rf results/scienceworld_smoke
python scripts/run_experiment.py --config configs/scienceworld_smoke.yaml --output-dir results
uv pip install -e ".[dev,scienceworld,alfworld,sentence_transformers,local_llm]"
export ALFWORLD_DATA=${HOME}/.cache/alfworld
export ALFWORLD_CONFIG=/Users/zhanwenchen/sqmem/config_alfworld.yaml
```

To run the experiment

```bash
/Users/zhanwenchen/llama.cpp/build/bin/llama-server \
  -m /Users/zhanwenchen/sqmem/qwen3.5_9b/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  --port 8080 \
  --ctx-size 8192 \
  -ngl 999 \
  --api-key local \
  --chat-template-kwargs '{"enable_thinking":false}'

kill $(cat logs/overnight.pid)
nohup ./scripts/run_overnight.sh > logs/overnight.log 2>&1 &
echo $! > logs/overnight.pid
disown

export ALFWORLD_DATA=/Users/zhanwenchen/.cache/alfworld
uv run python scripts/run_experiment.py \
  --config configs/r3_alfworld_st_llm.yaml \
  --run-id-prefix "$(date +%Y%m%d%H%M%S)_" \
  > logs/r3_alfworld_postfix.log 2>&1

```
