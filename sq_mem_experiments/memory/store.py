"""JSONL memory bank and return-transform controls for ablations."""
import json
import os
import random
from dataclasses import asdict

from sq_mem_experiments.schema import MemoryItem

VALID_TRANSFORMS = {"none", "zero", "shuffle", "reverse", "random_memory"}


class MemoryStore:
    def __init__(self, path: str):
        self.path = path
        self._cache: list[MemoryItem] | None = None

    def add(self, item: MemoryItem) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(item)) + "\n")
        if self._cache is not None:
            self._cache.append(item)

    def load_all(self) -> list[MemoryItem]:
        if not os.path.exists(self.path):
            return []
        items: list[MemoryItem] = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(MemoryItem(**json.loads(line)))
        return items

    def get_all(self, force_reload: bool = False) -> list[MemoryItem]:
        if self._cache is None or force_reload:
            self._cache = self.load_all()
        return self._cache

    def apply_return_transform(
        self, transform: str, seed: int = 42
    ) -> list[MemoryItem]:
        """Return a modified copy of all items for an ablation variant."""
        if transform not in VALID_TRANSFORMS:
            raise ValueError(f"Unknown transform '{transform}'. Valid: {VALID_TRANSFORMS}")
        # Shallow-copy each item so originals are unmodified
        items = [MemoryItem(**asdict(it)) for it in self.get_all()]
        if transform == "none":
            pass
        elif transform == "zero":
            for it in items:
                it.return_value = 0.0
        elif transform == "shuffle":
            rng = random.Random(seed)
            returns = [it.return_value for it in items]
            rng.shuffle(returns)
            for it, r in zip(items, returns):
                it.return_value = r
        elif transform == "reverse":
            if items:
                max_r = max(it.return_value for it in items)
                min_r = min(it.return_value for it in items)
                for it in items:
                    it.return_value = max_r + min_r - it.return_value
        elif transform == "random_memory":
            # Shuffle embedding vectors to break retrieval relevance
            rng = random.Random(seed)
            state_vecs = [it.state_vec for it in items]
            action_vecs = [it.action_vec for it in items]
            rng.shuffle(state_vecs)
            rng.shuffle(action_vecs)
            for it, sv, av in zip(items, state_vecs, action_vecs):
                it.state_vec = sv
                it.action_vec = av
        return items

    def summary(self) -> dict[str, object]:
        items = self.get_all()
        splits: dict[str, int] = {}
        for it in items:
            splits[it.split] = splits.get(it.split, 0) + 1
        task_ids = sorted({it.task_id for it in items})
        return {
            "total_items": len(items),
            "splits": splits,
            "task_ids": task_ids,
            "num_tasks": len(task_ids),
        }
