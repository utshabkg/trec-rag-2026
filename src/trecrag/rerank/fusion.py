"""Rank fusion over signals the cascade has already computed.

A cascade of rerankers keeps only the last one's opinion: each stage re-sorts by
its own score, so every earlier stage's ordering is discarded and only its
*truncation* survives. Measured on this pipeline, that made both LTR and QDER
byte-for-byte no-ops -- they reordered a pool that the cross-encoder then
re-sorted from scratch.

This stage makes earlier signals count. Every reranker writes its score into
`Candidate.features`, so by the time the cross-encoder has run, each candidate
carries `qder_score`, `ce_score`, `bm25_score`, `rrf_score` and so on. Fusing
those turns the cascade from "last scorer wins" into an ensemble.

Fusion is by RECIPROCAL RANK, not by score. Scores here are not comparable: the
cross-encoder emits logits that were all negative in a real run (max -1.84),
while an untrained QDER emits values on an arbitrary scale. Min-max normalising
them would let one stage's spread dominate for reasons unrelated to its
accuracy. RRF only needs each signal's ordering, which is the part we trust.

A weight of 0 disables a signal exactly, so this is safe to wire in before a
model is trained: the fusion is then equivalent to the single weighted stage.
"""

from __future__ import annotations

from trecrag.logging_utils import get_logger
from trecrag.rerank.base import Reranker
from trecrag.schema import Candidate, Topic

logger = get_logger("fusion")


class FeatureFusionReranker(Reranker):
    """Re-sort candidates by weighted RRF over several per-stage feature scores.

    `feature_weights` maps a feature name (as written into `Candidate.features`
    by an upstream stage) to its weight in the fusion. Candidates missing a
    feature simply do not receive a contribution from it, rather than being
    scored 0 -- a missing score means "this stage did not judge this candidate"
    (e.g. QDER skips text-less candidates), which is not the same as judging it
    worst.
    """

    name = "fusion"

    def __init__(
        self,
        feature_weights: dict[str, float],
        k_rrf: int = 60,
        strict: bool = True,
    ):
        self.feature_weights = {k: float(v) for k, v in (feature_weights or {}).items() if v}
        self.k_rrf = k_rrf
        self.strict = strict

    def rerank(self, topic: Topic, candidates: list[Candidate], k: int) -> list[Candidate]:
        if not candidates:
            return []
        if not self.feature_weights:
            return candidates[:k]

        fused: dict[int, float] = {i: 0.0 for i in range(len(candidates))}
        used: list[str] = []

        for feature, weight in self.feature_weights.items():
            scored = [i for i, c in enumerate(candidates) if feature in c.features]
            if not scored:
                # A weighted feature that no candidate carries means the stage
                # that produces it is disabled or renamed. Silently fusing
                # nothing would look identical to a working ensemble.
                message = (
                    f"fusion: no candidate has feature {feature!r}; "
                    f"is the stage that writes it enabled?"
                )
                if self.strict:
                    raise ValueError(message)
                logger.warning(message)
                continue

            used.append(feature)
            order = sorted(scored, key=lambda i: candidates[i].features[feature], reverse=True)
            for rank, i in enumerate(order, start=1):
                fused[i] += weight / (self.k_rrf + rank)

        if not used:
            return candidates[:k]

        ordered = sorted(range(len(candidates)), key=lambda i: fused[i], reverse=True)[:k]
        reranked: list[Candidate] = []
        for new_rank, i in enumerate(ordered, start=1):
            candidate = candidates[i]
            candidate.rank = new_rank
            candidate.score = fused[i]
            candidate.features["fusion_score"] = fused[i]
            reranked.append(candidate)
        return reranked
