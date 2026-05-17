"""Soft-Q Memory estimator: nonparametric value estimate from retrieved memories."""
from dataclasses import dataclass

import numpy as np

from sq_mem_experiments.memory.embeddings import Embedder, HashEmbedder
from sq_mem_experiments.schema import MemoryItem

WEIGHT_MODES = {"softmax", "uniform", "top1"}


@dataclass
class Retrieval:
    item: MemoryItem
    score: float
    weight: float


class SoftQMemory:
    """
    Estimates Q(s, a) = sum_i w_i * G_i over a soft retrieved neighborhood.

    retrieval score_i = alpha * sim(z_t, z_i) + (1-alpha) * sim(u, u_i)
    weights via softmax with temperature beta, or uniform/top-1 for ablations.
    """

    def __init__(
        self,
        items: list[MemoryItem],
        embedder: Embedder | None = None,
        top_r: int = 10,
        alpha: float = 0.5,
        beta: float = 0.1,
        action_conditioning: bool = True,
        weight_mode: str = "softmax",
    ):
        if weight_mode not in WEIGHT_MODES:
            raise ValueError(f"weight_mode must be one of {WEIGHT_MODES}")
        self.items = items
        self.embedder = embedder or HashEmbedder()
        self.top_r = top_r
        self.alpha = alpha
        self.beta = beta
        self.action_conditioning = action_conditioning
        self.weight_mode = weight_mode

        dim = self.embedder.dim
        if items:
            self._state_mat = np.array(
                [it.state_vec for it in items], dtype=np.float32
            )
            self._action_mat = np.array(
                [it.action_vec for it in items], dtype=np.float32
            )
            self._returns = np.array(
                [it.return_value for it in items], dtype=np.float32
            )
        else:
            self._state_mat = np.empty((0, dim), dtype=np.float32)
            self._action_mat = np.empty((0, dim), dtype=np.float32)
            self._returns = np.empty(0, dtype=np.float32)

    def estimate(
        self,
        state_text: str,
        action_text: str,
    ) -> tuple[float, float, list[Retrieval]]:
        """Return (q_value, uncertainty, retrievals)."""
        if len(self.items) == 0:
            return 0.0, 0.0, []

        z = self.embedder.embed(state_text)
        u = self.embedder.embed(action_text)

        state_sims = self._state_mat @ z
        if self.action_conditioning:
            action_sims = self._action_mat @ u
            scores = self.alpha * state_sims + (1.0 - self.alpha) * action_sims
        else:
            scores = state_sims.copy()

        k = min(self.top_r, len(self.items))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        top_scores = scores[top_idx]
        top_returns = self._returns[top_idx]

        if self.weight_mode == "top1":
            weights = np.zeros(k, dtype=np.float32)
            weights[0] = 1.0
        elif self.weight_mode == "uniform":
            weights = np.ones(k, dtype=np.float32) / k
        else:  # softmax
            shifted = top_scores - top_scores.max()
            exp_s = np.exp(shifted / max(self.beta, 1e-8))
            weights = exp_s / exp_s.sum()

        q = float(np.dot(weights, top_returns))
        variance = float(np.dot(weights, (top_returns - q) ** 2))
        uncertainty = float(np.sqrt(max(variance, 0.0)))

        retrievals = [
            Retrieval(
                item=self.items[int(idx)],
                score=float(scores[int(idx)]),
                weight=float(w),
            )
            for idx, w in zip(top_idx, weights)
        ]
        return q, uncertainty, retrievals

    def retrieval_similarity(self, state_text: str, action_text: str) -> float:
        """Average top-R retrieval score (used by semantic_retrieval variant)."""
        if len(self.items) == 0:
            return 0.0
        _, _, retrievals = self.estimate(state_text, action_text)
        if not retrievals:
            return 0.0
        return float(np.mean([r.score for r in retrievals]))

    def estimate_batch(
        self,
        state_text: str,
        action_texts: list[str],
    ) -> list[tuple[float, float, list[Retrieval]]]:
        """Batched estimate — embeds state once, actions in one call.

        Equivalent to calling estimate() once per action, but ~100x faster on
        sentence-transformer embedders because of GPU/Metal batching.
        """
        n = len(action_texts)
        if len(self.items) == 0 or n == 0:
            return [(0.0, 0.0, []) for _ in range(n)]

        z = self.embedder.embed(state_text)
        state_sims: np.ndarray = self._state_mat @ z

        action_sims_all = np.zeros((n, len(self.items)), dtype=np.float32)
        if self.action_conditioning:
            if hasattr(self.embedder, "embed_batch"):
                action_mat: np.ndarray = np.asarray(
                    self.embedder.embed_batch(action_texts),  # type: ignore[attr-defined]
                    dtype=np.float32,
                )
            else:
                action_mat = np.stack([self.embedder.embed(a) for a in action_texts])
            action_sims_all = action_mat @ self._action_mat.T  # (N_actions, N_items)

        results: list[tuple[float, float, list[Retrieval]]] = []
        k = min(self.top_r, len(self.items))
        for i in range(n):
            if self.action_conditioning:
                scores: np.ndarray = (
                    self.alpha * state_sims + (1.0 - self.alpha) * action_sims_all[i]
                )
            else:
                scores = state_sims

            top_idx = np.argpartition(scores, -k)[-k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            top_scores: np.ndarray = scores[top_idx]
            top_returns = self._returns[top_idx]

            if self.weight_mode == "top1":
                weights = np.zeros(k, dtype=np.float32)
                weights[0] = 1.0
            elif self.weight_mode == "uniform":
                weights = np.ones(k, dtype=np.float32) / k
            else:
                shifted = top_scores - float(top_scores.max())
                exp_s = np.exp(shifted / max(self.beta, 1e-8))
                weights = exp_s / exp_s.sum()

            q = float(np.dot(weights, top_returns))
            variance = float(np.dot(weights, (top_returns - q) ** 2))
            uncertainty = float(np.sqrt(max(variance, 0.0)))

            retrievals = [
                Retrieval(
                    item=self.items[int(idx)],
                    score=float(scores[int(idx)]),
                    weight=float(w),
                )
                for idx, w in zip(top_idx, weights)
            ]
            results.append((q, uncertainty, retrievals))
        return results
