"""Construct memory bank from completed training episodes."""
from sq_mem_experiments.memory.embeddings import Embedder
from sq_mem_experiments.memory.store import MemoryStore
from sq_mem_experiments.schema import Episode, MemoryItem, TaskSpec


def build_memory_from_episodes(
    episodes: list[Episode],
    store: MemoryStore,
    embedder: Embedder,
    allowed_task_ids: set[str],
) -> int:
    """Convert each training decision into a MemoryItem and write to store.

    Returns the number of items written.
    """
    counter = 0
    for episode in episodes:
        if episode.task_id not in allowed_task_ids:
            raise ValueError(
                f"Split violation: task_id '{episode.task_id}' is not in the "
                "allowed training set. Memory must only be built from train tasks."
            )
        for decision in episode.decisions:
            state_vec = embedder.embed_to_list(decision.state_text)
            action_vec = embedder.embed_to_list(decision.selected_action)
            item = MemoryItem(
                item_id=f"m_{counter:08d}",
                task_id=episode.task_id,
                split="train",
                step_index=decision.step_index,
                state_text=decision.state_text,
                action_text=decision.selected_action,
                action_name=decision.selected_action,
                return_value=decision.return_value,
                state_vec=state_vec,
                action_vec=action_vec,
                metadata={
                    "source_variant": episode.variant,
                    "episode_success": episode.success,
                    "reward": decision.reward,
                },
            )
            store.add(item)
            counter += 1
    return counter


def check_split_discipline(
    store: MemoryStore,
    test_tasks: list[TaskSpec],
) -> None:
    """Raise if any test task ID appears in the memory bank."""
    test_ids = {t.task_id for t in test_tasks}
    items = store.get_all()
    leaked = [it.task_id for it in items if it.task_id in test_ids]
    if leaked:
        raise RuntimeError(
            f"Split discipline violation: test task IDs found in memory bank: "
            f"{sorted(set(leaked))}"
        )
