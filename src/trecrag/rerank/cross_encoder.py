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


class LexicalOverlapScorer(PairScorer):
    def score(self, query: str, passage: str) -> float:
        query_tokens = set(tokenize(query))
        passage_tokens = tokenize(passage)
        if not query_tokens or not passage_tokens:
            return 0.0
        overlap = sum(1 for token in passage_tokens if token in query_tokens)
        return overlap / (len(query_tokens) ** 0.5 * len(passage_tokens) ** 0.5)


class SentenceTransformersCrossEncoderScorer(PairScorer):
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformersCrossEncoderScorer requires 'sentence-transformers': "
                "pip install trecrag[neural]"
            ) from exc
        self._model = CrossEncoder(model_name)

    def score(self, query: str, passage: str) -> float:
        return float(self._model.predict([(query, passage)])[0])


class CrossEncoderReranker(Reranker):
    name = "cross_encoder"

    def __init__(self, scorer: PairScorer):
        self.scorer = scorer

    def rerank(self, topic: Topic, candidates: list[Candidate], k: int) -> list[Candidate]:
        if not candidates:
            return []
        scores = [
            self.scorer.score(topic.text, candidate.text) if candidate.text else float("-inf")
            for candidate in candidates
        ]
        return _resorted(candidates, scores, "ce_score", k)
