"""Dense ("biencoder") side of the first-stage hybrid retriever.

ClimbMix is only exposed through the Lucene/BM25 Pyserini REST index — there
is no separately hosted dense ANN index. So the biencoder here re-scores the
BM25 candidate pool by cosine similarity in embedding space (a standard
approach when only a single sparse index is available); it is not an
independent recall source. `pool_size` controls how deep that pool goes.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np

from trecrag.retrieval.base import Retriever
from trecrag.schema import Candidate, Topic
from trecrag.text_utils import tokenize


class Encoder(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)


class HashingEncoder(Encoder):
    """Deterministic bag-of-words hashing-trick encoder. No model download,
    no GPU, no network — a stand-in dense encoder used until a real
    sentence-transformer checkpoint is configured."""

    def __init__(self, dims: int = 768):
        self.dims = dims

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dims), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest, "little") % self.dims
                sign = 1.0 if (digest[0] & 1) == 0 else -1.0
                vectors[i, index] += sign
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class SentenceTransformerEncoder(Encoder):
    """Real dense encoder backend (lazy-imported optional dependency).

    `query_prefix`/`passage_prefix` exist because the E5 family is trained with
    explicit "query: " / "passage: " markers and loses accuracy without them;
    they default to empty so prefix-free checkpoints are unaffected.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 128,
        query_prefix: str = "",
        passage_prefix: str = "",
        max_passage_chars: int = 2000,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerEncoder requires 'sentence-transformers': "
                "pip install trecrag[neural]"
            ) from exc
        self._model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.max_passage_chars = max_passage_chars

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode([f"{self.query_prefix}{t}" for t in texts])

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self.encode(
            [f"{self.passage_prefix}{t[: self.max_passage_chars]}" for t in texts]
        )


class BiencoderRetriever(Retriever):
    """Standalone use retrieves its own BM25 pool via `pool_retriever`. Inside
    the full pipeline, use `rescore()` directly on the BM25 candidates the
    pipeline already fetched, so BM25 search isn't issued twice per topic."""

    name = "biencoder"

    def __init__(
        self,
        encoder: Encoder,
        pool_retriever: Retriever | None = None,
        pool_size: int = 1000,
    ):
        self.encoder = encoder
        self.pool_retriever = pool_retriever
        self.pool_size = pool_size

    def retrieve(self, topic: Topic, k: int) -> list[Candidate]:
        if self.pool_retriever is None:
            raise ValueError("BiencoderRetriever.retrieve requires a pool_retriever")
        pool = self.pool_retriever.retrieve(topic, self.pool_size)
        return self.rescore(topic, pool, k)

    def rescore(self, topic: Topic, pool: list[Candidate], k: int) -> list[Candidate]:
        pool = [c for c in pool if c.text]
        if not pool:
            return []

        query_vec = self.encoder.encode_queries([topic.text])[0]
        doc_vecs = self.encoder.encode_passages([c.text for c in pool])
        scores = doc_vecs @ query_vec

        order = np.argsort(-scores)[:k]
        reranked: list[Candidate] = []
        for new_rank, idx in enumerate(order, start=1):
            source = pool[int(idx)]
            score = float(scores[int(idx)])
            features = dict(source.features)
            features["bien_score"] = score
            features["bien_rank"] = float(new_rank)
            reranked.append(
                Candidate(
                    docid=source.docid,
                    score=score,
                    rank=new_rank,
                    text=source.text,
                    features=features,
                )
            )
        return reranked
