"""Episode-level multiprocessing for SQ-Mem variant evaluation.

Workers persist across tasks: each worker process creates its own env,
embedder, and LLM policy once, then handles many tasks in sequence over
the pool's lifetime. JVM startup cost is paid N_WORKERS times per pool,
not N_TASKS times.

Default config (`agent.n_workers = 1`) takes the sequential fallback in
runner.py; no behavior change unless opt-in via YAML.

Determinism: our per-task RNG seeds with `hash((task_id, variant, "explore"))`
in `ScienceWorldSQMemAgent.reset()`, so episode results are fully
task-deterministic. Parallel execution order does not affect outputs;
the returned Episode list matches `pool.map`'s preserved input order.
"""
from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from sq_mem_experiments.schema import Episode, MemoryItem, TaskSpec


# Module-global state populated inside each worker process by _worker_init.
# The parent process never reads these — they live in worker memory only.
_WORKER: dict[str, Any] = {}


def _build_embedder(cfg: dict[str, Any]) -> Any:
    """Construct embedder from cfg — mirrors runner.py's setup path."""
    from sq_mem_experiments.memory.embeddings import HashEmbedder
    mem_cfg = cfg.get("memory", {})
    if mem_cfg.get("embedder", "hash") == "sentence_transformers":
        from sq_mem_experiments.memory.embeddings import STEmbedder
        return STEmbedder(
            model_name=str(mem_cfg.get("st_model", "all-MiniLM-L6-v2")),
        )
    return HashEmbedder(
        vocab_size=int(mem_cfg.get("vocab_size", 4096)),
        dim=int(mem_cfg.get("dim", 64)),
    )


def _build_agent_config(cfg: dict[str, Any]) -> Any:
    """Construct AgentConfig from cfg — mirrors runner.py's setup path."""
    from sq_mem_experiments.agents.scienceworld_agents import AgentConfig
    agent_cfg = cfg.get("agent", {})
    mem_cfg = cfg.get("memory", {})
    return AgentConfig(
        lambda_memory=float(agent_cfg.get("lambda_memory", 1.0)),
        rho_uncertainty=float(agent_cfg.get("rho_uncertainty", 0.5)),
        top_r=int(mem_cfg.get("top_r", 10)),
        alpha=float(mem_cfg.get("alpha", 0.5)),
        beta=float(mem_cfg.get("beta", 0.1)),
        epsilon=float(agent_cfg.get("epsilon", 0.0)),
        memory_mode=str(agent_cfg.get("memory_mode", "q_rerank")),
        rag_top_k=int(agent_cfg.get("rag_top_k", 5)),
        normalize_actions=bool(agent_cfg.get("normalize_actions", False)),
        state_value_in_prompt=bool(agent_cfg.get("state_value_in_prompt", False)),
        q_in_prompt=bool(agent_cfg.get("q_in_prompt", False)),
    )


def _build_llm_policy(cfg: dict[str, Any]) -> Any:
    """Construct LLM policy if enabled in cfg, else None."""
    llm_cfg = cfg.get("agent", {}).get("llm", {})
    if not llm_cfg.get("enabled", False):
        return None
    from sq_mem_experiments.agents.llm_policy import LLMBasePolicy
    return LLMBasePolicy(
        model=str(llm_cfg.get("model", "claude-haiku-4-5")),
        max_candidates=int(llm_cfg.get("max_candidates", 20)),
        provider=str(llm_cfg.get("provider", "anthropic")),
        url=str(llm_cfg.get("url", "http://localhost:11434/v1")),
        repetition_window=int(llm_cfg.get("repetition_window", 3)),
    )


def _worker_init(cfg_pickled: bytes) -> None:
    """One-time setup per worker process.

    Builds env + embedder + agent_config + llm_policy once; stored in
    module-global _WORKER. Subsequent _worker_rollout calls reuse these.
    """
    from sq_mem_experiments.evaluation.runner import _make_env

    cfg = pickle.loads(cfg_pickled)
    _WORKER["cfg"] = cfg
    _WORKER["env"] = _make_env(cfg, generate_gold_path=False)
    _WORKER["embedder"] = _build_embedder(cfg)
    _WORKER["agent_cfg"] = _build_agent_config(cfg)
    _WORKER["llm_policy"] = _build_llm_policy(cfg)


def _worker_rollout(args: tuple[str, TaskSpec, bytes, int, float]) -> Episode:
    """Run one episode in the worker's env, return the populated Episode."""
    from sq_mem_experiments.agents.scienceworld_agents import make_agent
    from sq_mem_experiments.evaluation.rollout import rollout_episode

    variant, task_spec, memory_items_pickled, max_steps, success_threshold = args
    memory_items: list[MemoryItem] = pickle.loads(memory_items_pickled)

    agent = make_agent(
        variant, memory_items,
        _WORKER["embedder"],
        _WORKER["agent_cfg"],
        _WORKER["llm_policy"],
    )
    ep = rollout_episode(
        agent, _WORKER["env"], task_spec,
        max_steps=max_steps,
        success_threshold=success_threshold,
    )
    ep.variant = variant
    return ep


def rollout_variant_parallel(
    variant: str,
    test_tasks: list[TaskSpec],
    memory_items: list[MemoryItem],
    cfg: dict[str, Any],
    max_steps: int,
    success_threshold: float,
    n_workers: int = 4,
) -> list[Episode]:
    """Run all `test_tasks` for `variant` across `n_workers` worker processes.

    Each worker creates its own env once; tasks are distributed via
    `pool.map` (order-preserving). Returned episodes align with `test_tasks`.

    `memory_items` is pickled once and sent per task — for ~2000-item banks
    this is ~5MB per call, negligible vs the per-step env cost.
    """
    if n_workers <= 1:
        raise ValueError(
            "rollout_variant_parallel requires n_workers >= 2; "
            "use the sequential path in runner.py for n_workers == 1"
        )
    cfg_pickled = pickle.dumps(cfg)
    mem_pickled = pickle.dumps(memory_items)
    args_list = [
        (variant, task, mem_pickled, max_steps, success_threshold)
        for task in test_tasks
    ]
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(cfg_pickled,),
    ) as pool:
        return list(pool.map(_worker_rollout, args_list))
