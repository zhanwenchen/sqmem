"""Unit tests for action-conditioned retrieval and split-discipline checks."""
import math
import pathlib

import pytest

from sq_mem_experiments.evaluation.memory_builder import check_split_discipline
from sq_mem_experiments.memory.embeddings import HashEmbedder
from sq_mem_experiments.memory.soft_q_memory import SoftQMemory
from sq_mem_experiments.memory.store import MemoryStore
from sq_mem_experiments.schema import MemoryItem, TaskSpec


def _make_item(
    item_id: str,
    state: str,
    action: str,
    ret: float,
    embedder: HashEmbedder,
    task_id: str = "train_task_0",
    split: str = "train",
) -> MemoryItem:
    return MemoryItem(
        item_id=item_id,
        task_id=task_id,
        split=split,
        step_index=0,
        state_text=state,
        action_text=action,
        action_name=action,
        return_value=ret,
        state_vec=embedder.embed_to_list(state),
        action_vec=embedder.embed_to_list(action),
    )


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(vocab_size=512, dim=32, seed=0)


# ---------------------------------------------------------------------------
# Action-conditioned retrieval
# ---------------------------------------------------------------------------

def test_action_conditioning_discriminates_actions(embedder: HashEmbedder) -> None:
    """Same state, two different actions with different returns.

    With action conditioning, querying one action should yield a higher Q than
    querying the other.
    """
    state = "water beaker stove heating task"
    good_action = "heat water stove fire boil"
    bad_action = "pick rock table wrong direction"

    items = [
        _make_item("m_0", state, good_action, 1.0, embedder),
        _make_item("m_1", state, good_action, 0.9, embedder),
        _make_item("m_2", state, bad_action, -0.5, embedder),
        _make_item("m_3", state, bad_action, -0.6, embedder),
    ]

    sqm = SoftQMemory(items, embedder=embedder, top_r=4, action_conditioning=True)
    q_good, _, _ = sqm.estimate(state, good_action)
    q_bad, _, _ = sqm.estimate(state, bad_action)
    assert q_good > q_bad, (
        f"Action conditioning should prefer good_action (Q={q_good:.3f}) "
        f"over bad_action (Q={q_bad:.3f})"
    )


def test_no_action_conditioning_ignores_action(embedder: HashEmbedder) -> None:
    """Without action conditioning, the same state query yields the same Q regardless of action."""
    state = "same state text here"
    items = [
        _make_item("m_0", state, "action one", 1.0, embedder),
        _make_item("m_1", state, "action two", -1.0, embedder),
    ]
    sqm = SoftQMemory(items, embedder=embedder, top_r=2, action_conditioning=False)
    q1, _, _ = sqm.estimate(state, "action one")
    q2, _, _ = sqm.estimate(state, "action two")
    assert math.isclose(q1, q2, rel_tol=1e-5), (
        "Without action conditioning, Q should be the same for any action"
    )


# ---------------------------------------------------------------------------
# Retrieval relevance
# ---------------------------------------------------------------------------

def test_retrieval_score_is_higher_for_similar_state(embedder: HashEmbedder) -> None:
    items = [
        _make_item("m_0", "boil water beaker stove fire heat", "heat water", 0.9, embedder),
        _make_item("m_1", "completely different context xyz", "unrelated act", 0.1, embedder),
    ]
    sqm = SoftQMemory(items, embedder=embedder, top_r=2)
    _, _, retrievals = sqm.estimate("boil water beaker stove", "heat water")
    scores = {r.item.item_id: r.score for r in retrievals}
    assert scores["m_0"] > scores["m_1"]


# ---------------------------------------------------------------------------
# Split discipline
# ---------------------------------------------------------------------------

def test_check_split_discipline_passes_clean_memory(
    tmp_path: pathlib.Path, embedder: HashEmbedder
) -> None:
    store = MemoryStore(str(tmp_path / "bank.jsonl"))
    store.add(_make_item("m_0", "state", "action", 0.5, embedder, task_id="train_boil_var0000"))

    test_tasks = [
        TaskSpec("test_boil_var0020", "scienceworld", "boil", 20, "test")
    ]
    check_split_discipline(store, test_tasks)  # should not raise


def test_check_split_discipline_raises_on_leakage(
    tmp_path: pathlib.Path, embedder: HashEmbedder
) -> None:
    store = MemoryStore(str(tmp_path / "bank.jsonl"))
    store.add(_make_item(
        "m_0", "state", "action", 0.5, embedder,
        task_id="test_boil_var0020", split="test"
    ))

    test_tasks = [
        TaskSpec("test_boil_var0020", "scienceworld", "boil", 20, "test")
    ]
    with pytest.raises(RuntimeError, match="Split discipline violation"):
        check_split_discipline(store, test_tasks)


# ---------------------------------------------------------------------------
# Memory store CRUD
# ---------------------------------------------------------------------------

def test_store_add_and_load(
    tmp_path: pathlib.Path, embedder: HashEmbedder
) -> None:
    path = str(tmp_path / "bank.jsonl")
    store = MemoryStore(path)
    item = _make_item("m_0", "state", "action", 0.42, embedder)
    store.add(item)

    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].item_id == "m_0"
    assert loaded[0].return_value == pytest.approx(0.42)


def test_store_summary(
    tmp_path: pathlib.Path, embedder: HashEmbedder
) -> None:
    store = MemoryStore(str(tmp_path / "bank.jsonl"))
    store.add(_make_item("m_0", "s", "a", 0.1, embedder, task_id="train_t0"))
    store.add(_make_item("m_1", "s", "b", 0.2, embedder, task_id="train_t1"))
    summary = store.summary()
    assert summary["total_items"] == 2
    assert summary["num_tasks"] == 2
