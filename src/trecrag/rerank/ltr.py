"""Learning-to-rank fusion stage.

Consumes the numeric signals accumulated in `Candidate.features` by the BM25,
biencoder, and RRF stages (`bm25_score`, `bm25_rank`, `bien_score`,
`bien_rank`, `rrf_score`) and produces one LTR score per candidate.

Two backends:
  - a trained LightGBM LambdaMART booster (`model_path` in config), the
    intended production path once a model is trained (see
    `scripts/train_ltr.py`);
  - a dependency-free per-feature min-max-normalized weighted sum, used
    whenever no trained model is configured, so the stage is never a no-op.
"""

from __future__ import annotations

import numpy as np

from trecrag.rerank.base import Reranker, _resorted
from trecrag.schema import Candidate, Topic


class LTRReranker(Reranker):
    name = "ltr"

    def __init__(
        self,
        feature_names: list[str],
        weights: dict[str, float] | None = None,
        model_path: str | None = None,
    ):
        self.feature_names = feature_names
        self.weights = weights or {}
        self._booster = self._load_booster(model_path) if model_path else None

    @staticmethod
    def _load_booster(model_path: str):
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "LTRReranker with a model_path requires 'lightgbm': pip install trecrag[ltr]"
            ) from exc
        return lgb.Booster(model_file=model_path)

    def _feature_matrix(self, candidates: list[Candidate]) -> np.ndarray:
        rows = [[c.features.get(name, 0.0) for name in self.feature_names] for c in candidates]
        return np.asarray(rows, dtype=np.float32)

    def _linear_fallback(self, matrix: np.ndarray) -> list[float]:
        normalized = np.zeros_like(matrix)
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            lo, hi = float(col.min()), float(col.max())
            normalized[:, j] = (col - lo) / (hi - lo) if hi > lo else 0.0
        weight_vec = np.array(
            [self.weights.get(name, 0.0) for name in self.feature_names], dtype=np.float32
        )
        return list(normalized @ weight_vec)

    def rerank(self, topic: Topic, candidates: list[Candidate], k: int) -> list[Candidate]:
        if not candidates:
            return []
        matrix = self._feature_matrix(candidates)
        if self._booster is not None:
            scores = [float(s) for s in self._booster.predict(matrix)]
        else:
            scores = self._linear_fallback(matrix)
        return _resorted(candidates, scores, "ltr_score", k)
