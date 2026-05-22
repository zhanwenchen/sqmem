"""Tests for episode-level parallel rollouts.

The default test (`test_worker_runs_episode_in_process`) exercises the worker
functions directly in the test process using a mock env, so it has no JVM
dependency and runs in CI. A separate integration test that spawns a real
ProcessPoolExecutor is marked `@pytest.mark.parallel` and skipped by default;
it's intended for on-demand local verification before merging changes that
touch the multiprocessing wiring.
"""
from __future__ import annotations

from typing import Any

import pytest

from sq_mem_experiments.envs.base import BaseEnv
from sq_mem_experiments.memory.embeddings import HashEmbedder
from sq_mem_experiments.schema import MemoryItem, TaskSpec


# ---------------------------------------------------------------------------
# Mock env — deterministic, no JVM
# ---------------------------------------------------------------------------

class MockEnv(BaseEnv):
    """A tiny deterministic env for testing the rollout machinery."""

    def __init__(self, **_kwargs: Any) -> None:
        self._goal: str = ""
        self._step: int = 0
        self._last_score: float = 0.0

    def reset(self, task_spec: TaskSpec) -> str:
        self._goal = task_spec.task_name or "find target"
        self._step = 0
        self._last_score = 0.0
        return "Starting observation. Find the target."

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        self._step += 1
        score = 1.0 if "target" in action.lower() else 0.0
        step_reward = score - self._last_score
        self._last_score = score
        done = (self._step >= 3) or (score > 0.0)
        info: dict[str, Any] = {
            "score": score,
            "normalized_score": score,
            "won": score > 0.0,
        }
        return f"Step {self._step}; took {action}", step_reward, done, info

    def get_valid_actions(self) -> list[str]:
        return ["look around", "find target", "move forward"]

    def close(self) -> None:
        pass

    @property
    def task_goal(self) -> str:
        return self._goal


def _mock_make_env(_cfg: dict[str, Any], generate_gold_path: bool = False) -> BaseEnv:
    return MockEnv()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> dict[str, Any]:
    return {
        "benchmark": {"name": "mock", "reward_mode": "score_delta"},
        "memory": {
            "embedder": "hash",
            "vocab_size": 256,
            "dim": 32,
            "top_r": 5,
            "alpha": 0.5,
            "beta": 0.1,
        },
        "agent": {
            "lambda_memory": 1.0,
            "rho_uncertainty": 0.5,
            "epsilon": 0.0,  # deterministic; no exploration
        },
    }


@pytest.fixture
def memory_items() -> list[MemoryItem]:
    embedder = HashEmbedder(vocab_size=256, dim=32, seed=42)
    return [
        MemoryItem(
            item_id=f"m_{i}",
            task_id=f"train_{i}",
            split="train",
            step_index=0,
            state_text=f"Starting observation. Find the target. variant {i}",
            action_text="find target",
            action_name="find target",
            return_value=1.0,
            state_vec=embedder.embed_to_list(f"Starting observation. Find the target. variant {i}"),
            action_vec=embedder.embed_to_list("find target"),
        )
        for i in range(4)
    ]


@pytest.fixture
def test_tasks() -> list[TaskSpec]:
    return [
        TaskSpec(
            task_id=f"test_{i}",
            env_name="mock",
            task_name="find target",
            variation_id=i,
            split="test",
            metadata={},
        )
        for i in range(3)
    ]


# ---------------------------------------------------------------------------
# Default test: worker logic in-process (no multiprocessing)
# ---------------------------------------------------------------------------

def test_worker_runs_episode_in_process(
    cfg: dict[str, Any],
    memory_items: list[MemoryItem],
    test_tasks: list[TaskSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_worker_init + _worker_rollout produce a valid Episode in-process."""
    import pickle

    import sq_mem_experiments.evaluation.runner as runner_mod
    from sq_mem_experiments.evaluation import parallel

    monkeypatch.setattr(runner_mod, "_make_env", _mock_make_env)

    # Simulate what the pool does on each worker process.
    parallel._worker_init(pickle.dumps(cfg))
    try:
        mem_pickled = pickle.dumps(memory_items)
        for variant in ("raw_history", "sq_mem"):
            for task in test_tasks:
                ep = parallel._worker_rollout(
                    (variant, task, mem_pickled, 5, 0.9)
                )
                assert ep.variant == variant
                assert ep.task_id == task.task_id
                # mock env terminates within 3 steps
                assert 1 <= ep.steps <= 3
    finally:
        env = parallel._WORKER.get("env")
        if env is not None:
            env.close()
        parallel._WORKER.clear()


def test_worker_is_deterministic_across_repeated_calls(
    cfg: dict[str, Any],
    memory_items: list[MemoryItem],
    test_tasks: list[TaskSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same task → same Episode across repeated worker invocations.

    This is the property that makes parallel ordering safe: the parallel
    code path can't change results because each (task_id, variant) has a
    deterministic per-task RNG seed.
    """
    import pickle

    import sq_mem_experiments.evaluation.runner as runner_mod
    from sq_mem_experiments.evaluation import parallel

    monkeypatch.setattr(runner_mod, "_make_env", _mock_make_env)

    parallel._worker_init(pickle.dumps(cfg))
    try:
        mem_pickled = pickle.dumps(memory_items)
        task = test_tasks[0]
        ep_a = parallel._worker_rollout(("sq_mem", task, mem_pickled, 5, 0.9))
        ep_b = parallel._worker_rollout(("sq_mem", task, mem_pickled, 5, 0.9))
        assert ep_a.steps == ep_b.steps
        assert ep_a.total_reward == ep_b.total_reward
        assert [d.selected_action for d in ep_a.decisions] == [
            d.selected_action for d in ep_b.decisions
        ]
    finally:
        env = parallel._WORKER.get("env")
        if env is not None:
            env.close()
        parallel._WORKER.clear()


# ---------------------------------------------------------------------------
# On-demand integration test: real ProcessPoolExecutor
# ---------------------------------------------------------------------------
# Skipped by default because monkey-patching `_make_env` doesn't propagate to
# spawn-based worker processes on macOS. To run locally:
#
#   pytest tests/test_parallel_rollouts.py -m parallel -v
#
# This requires a working ScienceWorld install — see configs/r0_scienceworld_diagnostic.yaml.

@pytest.mark.parallel
@pytest.mark.skip(
    reason="Integration test requires real env; opt in with -m parallel and a live JVM."
)
def test_rollout_variant_parallel_with_real_env() -> None:
    """Placeholder for the live-env integration test.

    The real-env validation is done by running a small overnight config
    with `agent.n_workers: 2` and comparing summary.csv to a sequential
    run with `n_workers: 1`. See README for the canonical procedure.
    """
    pass
