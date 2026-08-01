"""Fuse the BM25 and biencoder ranked lists into one first-stage candidate set."""

from __future__ import annotations

from trecrag.schema import Candidate


def reciprocal_rank_fusion(
    ranked_lists: list[list[Candidate]],
    k_const: int = 60,
    top_k: int | None = None,
) -> list[Candidate]:
    """Standard RRF: fused_score(d) = sum over lists containing d of 1/(k_const + rank).

    Candidates are merged by `docid`; text and per-stage `features` are merged
    across lists (first non-null text wins, features are unioned) so
    downstream stages (LTR) see both `bm25_*` and `bien_*` signals.
    """
    merged: dict[str, Candidate] = {}
    fused_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for candidate in ranked_list:
            fused_scores[candidate.docid] = fused_scores.get(candidate.docid, 0.0) + 1.0 / (
                k_const + candidate.rank
            )
            if candidate.docid not in merged:
                merged[candidate.docid] = Candidate(
                    docid=candidate.docid,
                    score=0.0,
                    rank=0,
                    text=candidate.text,
                    features=dict(candidate.features),
                )
            else:
                existing = merged[candidate.docid]
                if not existing.text and candidate.text:
                    existing.text = candidate.text
                existing.features.update(candidate.features)

    ordered_docids = sorted(fused_scores, key=lambda docid: fused_scores[docid], reverse=True)
    if top_k is not None:
        ordered_docids = ordered_docids[:top_k]

    fused: list[Candidate] = []
    for new_rank, docid in enumerate(ordered_docids, start=1):
        candidate = merged[docid]
        candidate.score = fused_scores[docid]
        candidate.rank = new_rank
        candidate.features["rrf_score"] = fused_scores[docid]
        fused.append(candidate)
    return fused
