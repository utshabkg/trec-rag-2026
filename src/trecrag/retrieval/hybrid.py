"""Fuse the BM25 and biencoder ranked lists into one first-stage candidate set."""

from __future__ import annotations

from trecrag.schema import Candidate


def reciprocal_rank_fusion(
    ranked_lists: list[list[Candidate]],
    k_const: int = 60,
    top_k: int | None = None,
    weights: list[float] | None = None,
) -> list[Candidate]:
    """Standard RRF: fused_score(d) = sum over lists containing d of w/(k_const + rank).

    Candidates are merged by `docid`; text and per-stage `features` are merged
    across lists (first non-null text wins, features are unioned) so
    downstream stages (LTR) see both `bm25_*` and `bien_*` signals.

    `weights` scales each list's contribution and defaults to 1.0 everywhere,
    which is plain RRF. It exists for fusing lists that are not peers: when one
    whole-narrative run is fused against ~11 per-AIN runs, unweighted RRF gives
    the narrative a ~1/12 vote purely because it is outnumbered.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    elif len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights has {len(weights)} entries for {len(ranked_lists)} ranked lists"
        )

    merged: dict[str, Candidate] = {}
    fused_scores: dict[str, float] = {}

    for weight, ranked_list in zip(weights, ranked_lists):
        for candidate in ranked_list:
            fused_scores[candidate.docid] = fused_scores.get(candidate.docid, 0.0) + weight / (
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
