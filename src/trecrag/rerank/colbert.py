"""Late-interaction (ColBERT-style) reranking stage.

Real backend (`TransformerTokenEncoder`) uses contextual per-token hidden
states from any HF encoder checkpoint (e.g. a ColBERT/BERT checkpoint) and
scores with MaxSim. The dependency-free fallback (`HashingTokenEncoder`)
represents each token as a one-hot hashed vector, so MaxSim degenerates to a
lexical (token-overlap) late-interaction proxy — enough to keep the pipeline
runnable end to end without a GPU or model download.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np

from trecrag.rerank.base import Reranker, _resorted
from trecrag.schema import Candidate, Topic
from trecrag.text_utils import tokenize


class TokenEncoder(ABC):
    @abstractmethod
    def encode_tokens(self, text: str, max_tokens: int) -> np.ndarray:
        """Return an [T, dims] matrix of (approximately) unit-norm token vectors."""


class HashingTokenEncoder(TokenEncoder):
    def __init__(self, dims: int = 256):
        self.dims = dims

    def encode_tokens(self, text: str, max_tokens: int) -> np.ndarray:
        tokens = tokenize(text)[:max_tokens]
        if not tokens:
            return np.zeros((0, self.dims), dtype=np.float32)
        matrix = np.zeros((len(tokens), self.dims), dtype=np.float32)
        for i, token in enumerate(tokens):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "little") % self.dims
            matrix[i, index] = 1.0
        return matrix


class TransformerTokenEncoder(TokenEncoder):
    def __init__(self, model_name: str):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "TransformerTokenEncoder requires 'torch' and 'transformers': "
                "pip install trecrag[neural]"
            ) from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()

    def encode_tokens(self, text: str, max_tokens: int) -> np.ndarray:
        torch = self._torch
        inputs = self._tokenizer(
            text, truncation=True, max_length=max_tokens, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        embeddings = outputs.last_hidden_state[0]
        norms = embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return (embeddings / norms).numpy().astype(np.float32)


def max_sim(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    if query_tokens.shape[0] == 0 or doc_tokens.shape[0] == 0:
        return 0.0
    similarity = query_tokens @ doc_tokens.T
    return float(similarity.max(axis=1).sum())


class ColBERTReranker(Reranker):
    name = "colbert"

    def __init__(
        self,
        token_encoder: TokenEncoder,
        max_query_tokens: int = 32,
        max_doc_tokens: int = 180,
    ):
        self.token_encoder = token_encoder
        self.max_query_tokens = max_query_tokens
        self.max_doc_tokens = max_doc_tokens

    def rerank(self, topic: Topic, candidates: list[Candidate], k: int) -> list[Candidate]:
        if not candidates:
            return []
        query_matrix = self.token_encoder.encode_tokens(topic.text, self.max_query_tokens)
        scores = []
        for candidate in candidates:
            if not candidate.text:
                scores.append(float("-inf"))
                continue
            doc_matrix = self.token_encoder.encode_tokens(candidate.text, self.max_doc_tokens)
            scores.append(max_sim(query_matrix, doc_matrix))
        return _resorted(candidates, scores, "colbert_score", k)
