"""Cross-encoder reranking stage: scores each (query, passage) pair jointly.

Real backend uses `sentence_transformers.CrossEncoder`. The dependency-free
fallback scores pairs by normalized token overlap between query and passage —
a weak but legitimate joint-scoring proxy that keeps this stage functional
without a model download.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from trecrag.rerank.base import Reranker, _resorted
from trecrag.schema import Candidate, Topic
from trecrag.text_utils import tokenize


class PairScorer(ABC):
    @abstractmethod
    def score(self, query: str, passage: str) -> float: ...

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score many (query, passage) pairs at once.

        The default loops `score()`, which is what the dependency-free fallback
        wants. Model-backed scorers override this so a whole candidate list is
        one padded GPU forward pass instead of one pass per candidate.
        """
        return [self.score(query, passage) for query, passage in pairs]


class LexicalOverlapScorer(PairScorer):
    def score(self, query: str, passage: str) -> float:
        query_tokens = set(tokenize(query))
        passage_tokens = tokenize(passage)
        if not query_tokens or not passage_tokens:
            return 0.0
        overlap = sum(1 for token in passage_tokens if token in query_tokens)
        return overlap / (len(query_tokens) ** 0.5 * len(passage_tokens) ** 0.5)


class SentenceTransformersCrossEncoderScorer(PairScorer):
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 64,
        max_passage_chars: int = 2000,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformersCrossEncoderScorer requires 'sentence-transformers': "
                "pip install trecrag[neural]"
            ) from exc
        self._model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size
        self.max_passage_chars = max_passage_chars

    def score(self, query: str, passage: str) -> float:
        return self.score_batch([(query, passage)])[0]

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        truncated = [(q, p[: self.max_passage_chars]) for q, p in pairs]
        scores = self._model.predict(truncated, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]


class CrossEncoderReranker(Reranker):
    name = "cross_encoder"

    def __init__(self, scorer: PairScorer):
        self.scorer = scorer

    def rerank(self, topic: Topic, candidates: list[Candidate], k: int) -> list[Candidate]:
        if not candidates:
            return []
        scorable = [i for i, c in enumerate(candidates) if c.text]
        batch_scores = self.scorer.score_batch(
            [(topic.text, candidates[i].text) for i in scorable]
        )
        scores = [float("-inf")] * len(candidates)
        for i, score in zip(scorable, batch_scores):
            scores[i] = score
        return _resorted(candidates, scores, "ce_score", k)
