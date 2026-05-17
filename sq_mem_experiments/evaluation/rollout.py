"""Episode rollout and backward return-to-go assignment."""
from typing import TYPE_CHECKING, Any

from sq_mem_experiments.agents.base import BaseAgent
from sq_mem_experiments.envs.base import BaseEnv
from sq_mem_experiments.schema import CandidateAction, Decision, Episode, TaskSpec

if TYPE_CHECKING:
    from sq_mem_experiments.envs.scienceworld_adapter import ScienceWorldAdapter


def rollout_episode(
    agent: BaseAgent,
    env: BaseEnv,
    task_spec: TaskSpec,
    max_steps: int = 50,
    success_threshold: float = 0.9,
) -> Episode:
    """Run one episode and return an Episode with return-to-go filled in."""
    obs = env.reset(task_spec)
    agent.reset(task_spec.task_id, env.task_goal)

    decisions: list[Decision] = []
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}

    for _ in range(max_steps):
        valid_actions = env.get_valid_actions()
        action = agent.act(obs, valid_actions)

        # Collect decisions logged by the agent this step
        if hasattr(agent, "pop_decisions"):
            decisions.extend(agent.pop_decisions())  # type: ignore[attr-defined]

        obs, reward, done, info = env.step(action)
        agent.update(reward, done, info)

        if decisions:
            decisions[-1].reward = reward

        total_reward += reward
        if done:
            break

    # Flush any remaining buffered decisions
    if hasattr(agent, "pop_decisions"):
        decisions.extend(agent.pop_decisions())  # type: ignore[attr-defined]

    # Backward pass: assign return-to-go
    running = 0.0
    for d in reversed(decisions):
        running += d.reward
        d.return_value = running

    final_score = float(info.get("normalized_score", info.get("score", 0.0)))
    if "normalized_score" not in info and "score" in info:
        final_score = final_score / 100.0
    success = final_score >= success_threshold

    return Episode(
        task_id=task_spec.task_id,
        variant=agent.variant,
        decisions=decisions,
        success=success,
        total_reward=total_reward,
        steps=len(decisions),
        metadata={
            "final_score": final_score,
            "done": done,
            "task_name": task_spec.task_name,
            "variation_id": task_spec.variation_id,
            "split": task_spec.split,
        },
    )


def rollout_gold_episode(
    env: "ScienceWorldAdapter",
    task_spec: TaskSpec,
    success_threshold: float = 0.9,
) -> Episode:
    """Follow the gold action sequence and record decisions with return-to-go.

    The env must have been constructed with generate_gold_path=True.
    If a gold action is not in the valid-action set it is executed anyway
    (ScienceWorld accepts free-text); the score just won't advance.
    """
    obs = env.reset(task_spec)
    gold = env.gold_sequence

    decisions: list[Decision] = []
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}

    for step_idx, action in enumerate(gold):
        state_text = obs[:500]
        valid = env.get_valid_actions()
        candidate = CandidateAction(action_text=action, action_name=action)

        d = Decision(
            task_id=task_spec.task_id,
            step_index=step_idx,
            state_text=state_text,
            observation=obs,
            candidates=[candidate],
            base_selected_action=action,
            selected_action=action,
            memory_changed_decision=False,
        )

        obs, reward, done, info = env.step(action)
        d.reward = reward
        total_reward += reward
        decisions.append(d)
        if done:
            break

    # Backward pass: return-to-go
    running = 0.0
    for d in reversed(decisions):
        running += d.reward
        d.return_value = running

    final_score = float(info.get("normalized_score", info.get("score", 0.0)))
    if "normalized_score" not in info and "score" in info:
        final_score = final_score / 100.0
    success = final_score >= success_threshold

    return Episode(
        task_id=task_spec.task_id,
        variant="gold",
        decisions=decisions,
        success=success,
        total_reward=total_reward,
        steps=len(decisions),
        metadata={
            "final_score": final_score,
            "done": done,
            "task_name": task_spec.task_name,
            "variation_id": task_spec.variation_id,
            "split": task_spec.split,
        },
    )
