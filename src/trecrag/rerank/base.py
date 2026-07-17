from __future__ import annotations

from abc import ABC, abstractmethod

from trecrag.schema import Candidate, Topic


class Reranker(ABC):
    name: str = "reranker"

    @abstractmethod
    def rerank(self, topic: Topic, candidates: list[Candidate], k: int) -> list[Candidate]: ...


def _resorted(candidates: list[Candidate], scores: list[float], feature_key: str, k: int) -> list[Candidate]:
    """Shared helper: attach `feature_key` scores, sort desc, renumber rank, truncate to k."""
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)[:k]
    reranked = []
    for new_rank, i in enumerate(order, start=1):
        candidate = candidates[i]
        candidate.score = scores[i]
        candidate.rank = new_rank
        candidate.features[feature_key] = scores[i]
        reranked.append(candidate)
    return reranked
