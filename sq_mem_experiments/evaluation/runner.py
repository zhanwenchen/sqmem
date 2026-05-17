"""Experiment orchestration: 7-stage pipeline from config to hypothesis report."""
from __future__ import annotations

import csv
import json
import os
from typing import Any

import pandas as pd
from tqdm import tqdm

from sq_mem_experiments.agents.scienceworld_agents import (
    AgentConfig,
    ScienceWorldRandomMemoryBuilderAgent,
    _VARIANT_SPECS,
    make_agent,
)
from sq_mem_experiments.envs.scienceworld_adapter import ScienceWorldAdapter
from sq_mem_experiments.evaluation.hypothesis_testing import evaluate_and_write
from sq_mem_experiments.evaluation.memory_builder import (
    build_memory_from_episodes,
    check_split_discipline,
)
from sq_mem_experiments.evaluation.metrics import (
    compute_calibration,
    compute_horizon_buckets,
    compute_intervention_summary,
    compute_summary,
    episodes_to_decisions_rows,
)
from sq_mem_experiments.evaluation.rollout import rollout_episode, rollout_gold_episode
from sq_mem_experiments.memory.embeddings import Embedder, HashEmbedder
from sq_mem_experiments.memory.store import MemoryStore
from sq_mem_experiments.schema import Episode, TaskSpec


# ---------------------------------------------------------------------------
# Task creation
# ---------------------------------------------------------------------------

def make_scienceworld_tasks(
    cfg: dict[str, Any],
) -> tuple[list[TaskSpec], list[TaskSpec]]:
    b = cfg["benchmark"]
    train_vars: list[int] = b["train_variations"]
    test_vars: list[int] = b["test_variations"]
    overlap = set(train_vars) & set(test_vars)
    if overlap:
        raise ValueError(f"train/test variation overlap: {sorted(overlap)}")
    return ScienceWorldAdapter.make_tasks(b["task_names"], train_vars, test_vars)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _write_csv(path: str, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fn = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_jsonl(path: str, objs: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for obj in objs:
            f.write(json.dumps(obj) + "\n")


def _write_summary_csv(out_dir: str, all_episodes: dict[str, list[Episode]]) -> None:
    rows: list[dict[str, Any]] = []
    for variant, eps in all_episodes.items():
        row = compute_summary(eps)
        row["variant"] = variant
        rows.append(row)
    if rows:
        _write_csv(
            os.path.join(out_dir, "summary.csv"),
            rows,
            ["variant", "n_episodes", "success_rate", "avg_steps",
             "avg_total_reward", "repeated_failure_rate",
             "error_recovery_rate", "intervention_episode_rate"],
        )


def _write_interventions_csv(
    out_dir: str,
    all_episodes: dict[str, list[Episode]],
    baseline_episodes: list[Episode] | None,
) -> None:
    rows: list[dict[str, Any]] = []
    for variant, eps in all_episodes.items():
        row = compute_intervention_summary(eps, baseline_episodes)
        if row:
            row["variant"] = variant
            rows.append(row)
    if rows:
        _write_csv(
            os.path.join(out_dir, "summary_interventions.csv"),
            rows,
            ["variant", "intervention_rate", "intervention_episode_rate",
             "beneficial_rate", "harmful_rate", "n_changed_decisions", "n_decisions"],
        )


def _write_value_destruction_csv(
    out_dir: str, all_episodes: dict[str, list[Episode]]
) -> None:
    sq_sr = compute_summary(all_episodes.get("sq_mem", [])).get("success_rate")
    destruction_variants = [
        "semantic_retrieval", "sq_mem_no_returns", "sq_mem_zero_returns",
        "sq_mem_shuffled_returns", "sq_mem_value_reversed", "sq_mem_random_memory",
    ]
    rows: list[dict[str, Any]] = []
    for v in destruction_variants:
        if v not in all_episodes:
            continue
        sr = compute_summary(all_episodes[v]).get("success_rate")
        if sr is None:
            continue
        delta = (sq_sr - sr) if sq_sr is not None else float("nan")
        rows.append({"variant": v, "success_rate": sr,
                     "delta_vs_sq_mem": delta, "sq_mem_success_rate": sq_sr})
    if rows:
        _write_csv(
            os.path.join(out_dir, "summary_value_destruction.csv"),
            rows,
            ["variant", "success_rate", "delta_vs_sq_mem", "sq_mem_success_rate"],
        )


def _write_horizon_csv(
    out_dir: str, all_episodes: dict[str, list[Episode]]
) -> None:
    rows: list[dict[str, Any]] = []
    for variant, eps in all_episodes.items():
        for row in compute_horizon_buckets(eps):
            row["variant"] = variant
            rows.append(row)
    if rows:
        _write_csv(
            os.path.join(out_dir, "summary_horizon_buckets.csv"),
            rows,
            ["variant", "horizon_bucket", "min_steps", "max_steps",
             "n_episodes", "success_rate"],
        )


def _write_decisions_csv(
    out_dir: str, variant: str, episodes: list[Episode]
) -> None:
    rows = episodes_to_decisions_rows(episodes, variant)
    if rows:
        _write_csv(
            os.path.join(out_dir, f"decisions_{variant}.csv"),
            rows,
        )


def _write_decisions_all_csv(
    out_dir: str, all_episodes: dict[str, list[Episode]]
) -> None:
    all_rows: list[dict[str, Any]] = []
    for variant, eps in all_episodes.items():
        all_rows.extend(episodes_to_decisions_rows(eps, variant))
    if all_rows:
        _write_csv(os.path.join(out_dir, "decisions_all.csv"), all_rows)


def _write_calibration_csv(
    out_dir: str, variant: str, episodes: list[Episode]
) -> None:
    rows = compute_calibration(episodes)
    if rows:
        _write_csv(
            os.path.join(out_dir, f"calibration_{variant}.csv"),
            rows,
            ["q_bin_low", "q_bin_high", "q_bin_center", "n_decisions", "success_rate"],
        )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    def __init__(self, cfg: dict[str, Any], out_dir: str):
        self.cfg = cfg
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def run(self) -> None:
        cfg = self.cfg
        out_dir = self.out_dir

        # Persist resolved config
        import yaml  # type: ignore[import]
        with open(os.path.join(out_dir, "config_resolved.yaml"), "w") as f:
            yaml.dump(cfg, f)

        b = cfg["benchmark"]
        max_steps: int = b.get("max_steps", 50)
        success_threshold: float = b.get("success_threshold", 0.9)
        reward_mode: str = b.get("reward_mode", "score_delta")

        # Stage 1: create tasks
        print("[runner] Stage 1: creating tasks")
        train_tasks, test_tasks = make_scienceworld_tasks(cfg)
        train_ids = {t.task_id for t in train_tasks}
        print(f"  {len(train_tasks)} train tasks, {len(test_tasks)} test tasks")

        # Embedder (shared across all stages)
        mem_cfg = cfg.get("memory", {})
        embedder_type = str(mem_cfg.get("embedder", "hash"))
        if embedder_type == "sentence_transformers":
            from sq_mem_experiments.memory.embeddings import STEmbedder
            st_model = str(mem_cfg.get("st_model", "all-MiniLM-L6-v2"))
            print(f"[runner] Embedder: sentence-transformers ({st_model})")
            embedder: Embedder = STEmbedder(st_model)
        else:
            print("[runner] Embedder: hash (low quality — use embedder: sentence_transformers for paper runs)")
            embedder = HashEmbedder(
                vocab_size=int(mem_cfg.get("vocab_size", 4096)),
                dim=int(mem_cfg.get("dim", 256)),
            )

        # Stage 2-4: collect training trajectories and build memory
        memory_path = os.path.join(out_dir, "memory_bank.jsonl")
        store = MemoryStore(memory_path)

        mb_cfg = cfg.get("memory_builder", {})
        episodes_per_task: int = int(mb_cfg.get("episodes_per_task", 1))
        heuristic_prob: float = float(mb_cfg.get("heuristic_prob", 0.6))

        mem_agent_type: str = mb_cfg.get("agent", "scienceworld_random_memory_builder")
        # gold_and_random: one gold episode + N random episodes per task (contrastive signal)
        use_gold = mem_agent_type in ("scienceworld_gold", "scienceworld_gold_and_random")
        use_random = mem_agent_type in ("scienceworld_random_memory_builder",
                                        "scienceworld_gold_and_random")
        random_eps: int = int(mb_cfg.get("random_episodes_per_task", episodes_per_task))
        total_mem_eps = len(train_tasks) * ((1 if use_gold else 0) +
                                             (random_eps if use_random else 0))
        print(f"[runner] Stage 2-4: collecting {total_mem_eps} training episodes "
              f"(agent={mem_agent_type})")
        env = ScienceWorldAdapter(reward_mode=reward_mode, generate_gold_path=use_gold)
        mem_episodes: list[Episode] = []
        with tqdm(total=total_mem_eps, desc="memory collection", unit="ep") as pbar:
            for task in train_tasks:
                if use_gold:
                    ep = rollout_gold_episode(env, task,
                                              success_threshold=success_threshold)
                    mem_episodes.append(ep)
                    pbar.update(1)
                    pbar.set_postfix({"task": task.task_id[-20:], "steps": ep.steps,
                                      "score": f"{ep.metadata.get('final_score', 0):.2f}"})
                if use_random:
                    for seed in range(random_eps):
                        agent = ScienceWorldRandomMemoryBuilderAgent(
                            heuristic_prob=heuristic_prob, seed=seed
                        )
                        ep = rollout_episode(
                            agent, env, task,
                            max_steps=max_steps,
                            success_threshold=success_threshold,
                        )
                        mem_episodes.append(ep)
                        pbar.update(1)
                        pbar.set_postfix({"task": task.task_id[-20:], "steps": ep.steps,
                                          "score": f"{ep.metadata.get('final_score', 0):.2f}"})
        env.close()

        n_items = build_memory_from_episodes(mem_episodes, store, embedder, train_ids)
        print(f"  wrote {n_items} memory items")

        # Validate split discipline
        check_split_discipline(store, test_tasks)

        # Write memory summary
        summary_data = store.summary()
        with open(os.path.join(out_dir, "memory_summary.json"), "w") as f:
            json.dump(summary_data, f, indent=2)

        # LLM base policy (optional)
        agent_cfg = cfg.get("agent", {})
        llm_cfg = agent_cfg.get("llm", {})
        llm_policy = None
        if llm_cfg.get("enabled", False):
            from sq_mem_experiments.agents.llm_policy import LLMBasePolicy
            llm_policy = LLMBasePolicy(
                model=str(llm_cfg.get("model", "claude-haiku-4-5")),
                max_candidates=int(llm_cfg.get("max_candidates", 20)),
                provider=str(llm_cfg.get("provider", "anthropic")),
                url=str(llm_cfg.get("url", "http://localhost:11434/v1")),
            )
            print(f"[runner] LLM base policy: provider={llm_policy.provider}, "
                  f"model={llm_policy.model}, "
                  f"max_candidates={llm_policy.max_candidates}")

        # Agent base config
        base_agent_config = AgentConfig(
            lambda_memory=float(agent_cfg.get("lambda_memory", 1.0)),
            rho_uncertainty=float(agent_cfg.get("rho_uncertainty", 0.5)),
            top_r=int(mem_cfg.get("top_r", 10)),
            alpha=float(mem_cfg.get("alpha", 0.5)),
            beta=float(mem_cfg.get("beta", 0.1)),
            epsilon=float(agent_cfg.get("epsilon", 0.0)),
        )
        if base_agent_config.epsilon > 0:
            print(f"[runner] ε-greedy exploration: {base_agent_config.epsilon}")

        # Stages 5-6: run test variants
        variants: list[str] = cfg.get("variants", ["raw_history", "sq_mem"])
        all_episodes: dict[str, list[Episode]] = {}

        for variant in variants:
            print(f"[runner] Running variant: {variant}")
            transform, _ = _VARIANT_SPECS.get(variant, (None, {}))
            if transform is not None:
                memory_items = store.apply_return_transform(transform)
            else:
                memory_items = []

            test_episodes: list[Episode] = []
            env2 = ScienceWorldAdapter(reward_mode=reward_mode)
            with tqdm(total=len(test_tasks), desc=variant, unit="ep") as pbar:
                for task in test_tasks:
                    agent = make_agent(variant, memory_items, embedder, base_agent_config, llm_policy)
                    ep = rollout_episode(
                        agent, env2, task,
                        max_steps=max_steps,
                        success_threshold=success_threshold,
                    )
                    ep.variant = variant
                    test_episodes.append(ep)
                    pbar.update(1)
                    pbar.set_postfix({"steps": ep.steps, "success": ep.success})
            env2.close()

            all_episodes[variant] = test_episodes

            # Per-variant artifacts
            _write_decisions_csv(out_dir, variant, test_episodes)
            _write_calibration_csv(out_dir, variant, test_episodes)
            _write_jsonl(
                os.path.join(out_dir, f"episodes_{variant}.jsonl"),
                [e.to_dict() for e in test_episodes],
            )

        # Stage 7: write aggregate artifacts and evaluate hypotheses
        print("[runner] Stage 7: writing summary artifacts")
        baseline = all_episodes.get("raw_history")
        _write_summary_csv(out_dir, all_episodes)
        _write_interventions_csv(out_dir, all_episodes, baseline)
        _write_value_destruction_csv(out_dir, all_episodes)
        _write_horizon_csv(out_dir, all_episodes)
        _write_decisions_all_csv(out_dir, all_episodes)

        # README stub for the run
        with open(os.path.join(out_dir, "README.md"), "w") as f:
            f.write(f"# Run: {cfg.get('run_id', 'unnamed')}\n\n")
            f.write(f"Variants: {variants}\n\n")
            f.write("See `hypothesis_report.md` for automated claim evaluation.\n")

        hypothesis_thresholds: dict[str, Any] = cfg.get("hypothesis_tests", {})
        evaluate_and_write(out_dir, hypothesis_thresholds)
        print(f"[runner] Done. Results in: {out_dir}")


def run_from_config(config_path: str, out_dir: str) -> None:
    import yaml  # type: ignore[import]
    with open(config_path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    run_id: str = cfg.get("run_id", os.path.splitext(os.path.basename(config_path))[0])
    run_dir = os.path.join(out_dir, run_id)
    ExperimentRunner(cfg, run_dir).run()
