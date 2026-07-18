"""QDER reranking stage: Query-specific Document and Entity Representations.

See plans/qder-approach.md for the full approach this implements (vendored
from SIGIR2025-QDER, see `_vendor/`). QDER extends ColBERT-style late
interaction with two channels — text and entity — each combined via
query-guided attention, then Addition (complementarity) and Multiplication
(alignment) interactions, pooled and scaled by an upstream retrieval score,
then scored with a learned bilinear projection. The final score is optionally
interpolated with the upstream BM25-like score (§2.7 of the approach doc).

Two backends:
  - `HashingQDERScorer` (default): the same architecture, but with hashed
    token/entity vectors instead of BERT + Wikipedia2Vec, and an untrained
    (identity) bilinear matrix — needs no GPU, model download, or entity
    linker. Consistent with every other stage's dependency-free fallback.
  - `TransformerQDERScorer`: wraps the vendored `QDERModel` (real BERT text
    channel); the entity channel is off by default (`NullEntityProvider`)
    since no WAT-linked Wikipedia2Vec table exists for ClimbMix — pass a real
    `EntityEmbeddingProvider` to enable it. To use a checkpoint trained with
    SIGIR2025-QDER's own `src/doc_ranking/train.py`, pass `model_path`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from trecrag.rerank.base import Reranker, _resorted
from trecrag.rerank.colbert import HashingTokenEncoder, TokenEncoder
from trecrag.rerank.qder.entities import EntityEmbeddingProvider, NullEntityProvider
from trecrag.schema import Candidate, Topic

_NEG_INF = float("-inf")


def _minmax_normalize(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class QDERScorer(ABC):
    @abstractmethod
    def score_batch(
        self, query_text: str, doc_texts: list[str], doc_scores: list[float]
    ) -> list[float]: ...


class HashingQDERScorer(QDERScorer):
    def __init__(
        self,
        text_encoder: TokenEncoder | None = None,
        entity_provider: EntityEmbeddingProvider | None = None,
        max_query_tokens: int = 32,
        max_doc_tokens: int = 256,
        max_entities: int = 8,
    ):
        self.text_encoder = text_encoder or HashingTokenEncoder(dims=128)
        self.entity_provider = entity_provider or NullEntityProvider()
        self.max_query_tokens = max_query_tokens
        self.max_doc_tokens = max_doc_tokens
        self.max_entities = max_entities

    def score_batch(
        self, query_text: str, doc_texts: list[str], doc_scores: list[float]
    ) -> list[float]:
        return [
            self._score_one(query_text, doc_text, doc_score)
            for doc_text, doc_score in zip(doc_texts, doc_scores)
        ]

    def _score_one(self, query_text: str, doc_text: str, doc_score: float) -> float:
        query_tokens = self.text_encoder.encode_tokens(query_text, self.max_query_tokens)
        doc_tokens = self.text_encoder.encode_tokens(doc_text, self.max_doc_tokens)
        pooled = self._channel_features(query_tokens, doc_tokens, doc_score)

        query_entities = self.entity_provider.get_entity_vectors(query_text, self.max_entities)
        doc_entities = self.entity_provider.get_entity_vectors(doc_text, self.max_entities)
        pooled += self._channel_features(query_entities, doc_entities, doc_score)

        if not pooled:
            return 0.0
        h = np.concatenate(pooled)
        return float(h @ h)  # bilinear scoring with an untrained (identity) M

    @staticmethod
    def _channel_features(
        query_vecs: np.ndarray, doc_vecs: np.ndarray, doc_score: float
    ) -> list[np.ndarray]:
        if query_vecs.shape[0] == 0 or doc_vecs.shape[0] == 0:
            return []
        attention = query_vecs @ doc_vecs.T
        attention = attention - attention.max(axis=1, keepdims=True)
        weights = np.exp(attention)
        weights = weights / weights.sum(axis=1, keepdims=True)
        weighted_doc = weights @ doc_vecs

        added = query_vecs + weighted_doc
        multiplied = query_vecs * weighted_doc

        return [multiplied.mean(axis=0) * doc_score, added.mean(axis=0) * doc_score]


class TransformerQDERScorer(QDERScorer):
    """Real backend: wraps the vendored `QDERModel` (BERT text channel).

    Scores one document at a time rather than batching across a topic's
    candidates — simpler and correct, at the cost of some throughput; batch
    it yourself if this becomes a bottleneck for a large `top_k`.
    """

    def __init__(
        self,
        model_name: str,
        use_entities: bool = False,
        entity_provider: EntityEmbeddingProvider | None = None,
        score_method: str = "bilinear",
        model_path: str | None = None,
        max_query_tokens: int = 32,
        max_doc_tokens: int = 256,
        max_entities: int = 8,
        device: str = "cpu",
    ):
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "TransformerQDERScorer requires 'torch' and 'transformers': "
                "pip install trecrag[neural]"
            ) from exc
        from trecrag.rerank.qder._vendor.qder_model import QDERModel

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.entity_provider = entity_provider or NullEntityProvider()
        self.max_query_tokens = max_query_tokens
        self.max_doc_tokens = max_doc_tokens
        self.max_entities = max_entities
        self.device = device

        self.model = QDERModel(
            pretrained=model_name,
            use_scores=True,
            use_entities=use_entities,
            score_method=score_method,
        ).to(device)
        if model_path:
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def _encode_text(self, text: str, max_len: int):
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )
        return (
            encoded["input_ids"].to(self.device),
            encoded["attention_mask"].to(self.device),
            encoded["token_type_ids"].to(self.device),
        )

    def _encode_entities(self, text: str):
        torch = self._torch
        vectors = self.entity_provider.get_entity_vectors(text, self.max_entities)
        if vectors.shape[0] == 0:
            return (
                torch.zeros(1, 1, self.entity_provider.dims, device=self.device),
                torch.zeros(1, 1, dtype=torch.bool, device=self.device),
            )
        tensor = torch.as_tensor(vectors, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.ones(1, vectors.shape[0], dtype=torch.bool, device=self.device)
        return tensor, mask

    def score_batch(
        self, query_text: str, doc_texts: list[str], doc_scores: list[float]
    ) -> list[float]:
        torch = self._torch
        q_ids, q_mask, q_type = self._encode_text(query_text, self.max_query_tokens)
        q_entity_emb, q_entity_mask = self._encode_entities(query_text)

        scores = []
        with torch.no_grad():
            for doc_text, doc_score in zip(doc_texts, doc_scores):
                d_ids, d_mask, d_type = self._encode_text(doc_text, self.max_doc_tokens)
                d_entity_emb, d_entity_mask = self._encode_entities(doc_text)
                output = self.model(
                    query_input_ids=q_ids,
                    query_attention_mask=q_mask,
                    query_token_type_ids=q_type,
                    query_entity_emb=q_entity_emb,
                    doc_input_ids=d_ids,
                    doc_attention_mask=d_mask,
                    doc_token_type_ids=d_type,
                    doc_entity_emb=d_entity_emb,
                    query_entity_mask=q_entity_mask,
                    doc_entity_mask=d_entity_mask,
                    doc_score=torch.tensor([doc_score], dtype=torch.float32, device=self.device),
                )
                scores.append(float(output["score"].item()))
        return scores


class QDERReranker(Reranker):
    """Pipeline-facing QDER stage.

    `bm25_feature_name` selects which upstream `Candidate.features` entry is
    used both as QDER's §2.4 relevance-signal scaling factor `s` and as the
    §2.7 hybrid-interpolation BM25 term. `hybrid_lambda` is a fixed stand-in
    for the paper's coordinate-ascent-tuned λ (no training infrastructure for
    that is wired up here); set it to `None` to disable interpolation and use
    the raw QDER score.
    """

    name = "qder"

    def __init__(
        self,
        scorer: QDERScorer,
        bm25_feature_name: str = "bm25_score",
        hybrid_lambda: float | None = 0.5,
    ):
        self.scorer = scorer
        self.bm25_feature_name = bm25_feature_name
        self.hybrid_lambda = hybrid_lambda

    def rerank(self, topic: Topic, candidates: list[Candidate], k: int) -> list[Candidate]:
        if not candidates:
            return []

        bm25_like = [c.features.get(self.bm25_feature_name, 0.0) for c in candidates]
        doc_scores = _minmax_normalize(bm25_like)
        doc_texts = [c.text or "" for c in candidates]

        raw_scores = self.scorer.score_batch(topic.text, doc_texts, doc_scores)
        qder_scores = [
            score if candidate.text else _NEG_INF
            for candidate, score in zip(candidates, raw_scores)
        ]

        for candidate, qder_score in zip(candidates, qder_scores):
            if qder_score != _NEG_INF:
                candidate.features["qder_score"] = qder_score

        final_scores = qder_scores
        if self.hybrid_lambda is not None:
            final_scores = self._hybrid_scores(bm25_like, qder_scores)

        return _resorted(candidates, final_scores, "qder_hybrid_score", k)

    def _hybrid_scores(self, bm25_like: list[float], qder_scores: list[float]) -> list[float]:
        bm25_norm = _minmax_normalize(bm25_like)
        scored_indices = [i for i, s in enumerate(qder_scores) if s != _NEG_INF]
        qder_norm_by_index = {}
        if scored_indices:
            normalized = _minmax_normalize([qder_scores[i] for i in scored_indices])
            qder_norm_by_index = dict(zip(scored_indices, normalized))

        combined = []
        for i, qder_score in enumerate(qder_scores):
            if qder_score == _NEG_INF:
                combined.append(_NEG_INF)
            else:
                combined.append(
                    self.hybrid_lambda * bm25_norm[i]
                    + (1 - self.hybrid_lambda) * qder_norm_by_index[i]
                )
        return combined
