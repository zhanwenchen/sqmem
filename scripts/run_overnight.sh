#!/bin/bash
# Overnight queue: full slate, heuristic first → LLM later, SW before ALFWorld.
#
#   Stage 1 — heuristic baselines (no llama-server needed, ~2 hr total):
#     1.  r1 ScienceWorld (hash + heuristic)         ~12 min
#     2.  r2 ScienceWorld (ST + heuristic)           ~35 min
#     3.  r1 ALFWorld    (hash + heuristic)          ~30 min
#     4.  r2 ALFWorld    (ST + heuristic)            ~30 min
#
#   Stage 2 — LLM runs (require Qwen3.5-9B on llama-server with the corrected
#             flags: --ctx-size 8192 -ngl 999 --chat-template-kwargs
#             '{"enable_thinking":false}'):
#     5.  r0 ScienceWorld diagnostic (LLM sanity)    ~25-30 min
#     6.  r0 ALFWorld diagnostic (LLM sanity)        ~50 min
#     7.  r3 ScienceWorld (LLM, 16 variants)         ~5-7 hr
#     8.  r3 ALFWorld    (LLM, 16 variants)          ~12-13 hr (max_steps=50)
#
#   Total wall time:                                 ~20-23 hr
#
# Order rationale:
#   - Heuristic baselines first: they don't need llama-server, so a broken
#     server doesn't waste this hour. They also catch any code regression in
#     the variant ablations cheaply.
#   - r0 SW diagnostic before r0 ALFWorld: SW is the known-good benchmark,
#     so an LLM-regime regression shows up there first. SW r0 is also faster
#     (~25 min vs ~50 min) — quicker signal for the same diagnostic intent.
#   - Both r0 diagnostics before any r3: if either r0 produces garbage, we
#     know not to trust the long r3 runs without re-investigating.
#   - SW r3 before ALFWorld r3: SW is the known-good benchmark; if anything
#     is wrong with the LLM regime it shows up there first. SW also has shorter
#     per-episode wall time, so finishes sooner.
#
# A failing run does NOT abort the queue — we want partial artifacts in the
# morning regardless. Per-run output goes to logs/<run_id>.log.
#
# Usage:
#   chmod +x scripts/run_overnight.sh
#   caffeinate -i nohup ./scripts/run_overnight.sh > logs/overnight.log 2>&1 &
#   disown
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

# Single batch timestamp shared by every run in this queue. Result dirs and
# log files both inherit this prefix so it's trivial to identify which runs
# belong to the same overnight session.
BATCH_TS="$(date +%Y%m%d%H%M%S)"
echo "[overnight] batch timestamp: ${BATCH_TS}"
echo

run_config() {
    local cfg="$1"
    local name
    name="$(basename "$cfg" .yaml)"
    local log="logs/${BATCH_TS}_${name}.log"
    echo "[overnight] $(date): START ${BATCH_TS}_${name}"
    echo "[overnight]   config: $cfg"
    echo "[overnight]   log:    $log"
    if uv run python scripts/run_experiment.py \
            --config "$cfg" \
            --run-id-prefix "${BATCH_TS}_" \
            > "$log" 2>&1; then
        echo "[overnight] $(date): DONE  ${BATCH_TS}_${name} (exit 0)"
    else
        echo "[overnight] $(date): FAIL  ${BATCH_TS}_${name} (exit $?) — continuing to next run"
    fi
    echo
}

check_llama_server() {
    if curl -sf http://localhost:8080/v1/models > /dev/null; then
        return 0
    fi
    return 1
}

# Stage 1 — heuristic baselines (no llama-server dependency, ~2 hr total)
echo "[overnight] $(date): Stage 1 — heuristic baselines"
run_config configs/r1_scienceworld_hash_heuristic.yaml
run_config configs/r2_scienceworld_st_heuristic.yaml
run_config configs/r1_alfworld_hash_heuristic.yaml
run_config configs/r2_alfworld_st_heuristic.yaml

# Stage 2 — LLM runs (Qwen3.5-9B on llama-server with corrected flags)
echo "[overnight] $(date): Stage 2 — LLM runs (checking llama-server first)"
if check_llama_server; then
    echo "[overnight] llama-server OK."
    echo

    run_config configs/r0_scienceworld_diagnostic.yaml
    if ! check_llama_server; then
        echo "[overnight] WARN: llama-server died after r0 SW. Skipping remaining LLM runs."
    else
        run_config configs/r0_alfworld_diagnostic.yaml
        if ! check_llama_server; then
            echo "[overnight] WARN: llama-server died after r0 AW. Skipping r3 stages."
        else
            run_config configs/r3_scienceworld_st_llm.yaml
            if ! check_llama_server; then
                echo "[overnight] WARN: llama-server died after r3 SW. Skipping r3 ALFWorld."
            else
                run_config configs/r3_alfworld_st_llm.yaml
            fi
        fi
    fi
else
    echo "[overnight] WARN: llama-server is not responding on :8080."
    echo "[overnight]       Skipping all Stage 2 LLM runs."
fi

echo "[overnight] $(date): all queued runs complete (batch ${BATCH_TS})."
echo "[overnight] Per-variant summaries:"
for d in "results/${BATCH_TS}_r1_scienceworld_hash_heuristic" \
         "results/${BATCH_TS}_r2_scienceworld_st_heuristic" \
         "results/${BATCH_TS}_r1_alfworld_hash_heuristic" \
         "results/${BATCH_TS}_r2_alfworld_st_heuristic" \
         "results/${BATCH_TS}_r0_scienceworld_diagnostic" \
         "results/${BATCH_TS}_r0_alfworld_diagnostic" \
         "results/${BATCH_TS}_r3_scienceworld_st_llm" \
         "results/${BATCH_TS}_r3_alfworld_st_llm"; do
    if [ -f "$d/summary.csv" ]; then
        echo
        echo "===== $d/summary.csv ====="
        cat "$d/summary.csv"
    fi
done
