"""Fuse per-AIN retrieval runs (and the composite run) into one evidence pool.

The arithmetic is the existing `reciprocal_rank_fusion` — it already merges by
docid, unions `features`, and keeps the first non-null `text`, which is exactly
what pooling needs. What this module adds is the provenance RRF cannot know:
which needs found a document, which sub-questions those needs belong to, and
whether the whole-narrative query found it.

That provenance is not decoration. Fusion collapses ~11 ranked lists into one,
and once collapsed there is no way to ask "does this pool actually cover every
part of the narrative, or is it ten documents about the first sub-question?" —
which is precisely the coverage failure decomposition exists to fix. The
per-sub-question features recorded here are what `evidence.py` later selects on.
"""

from __future__ import annotations

from dataclasses import dataclass

from trecrag.retrieval.hybrid import reciprocal_rank_fusion
from trecrag.schema import Candidate


@dataclass(frozen=True)
class AinRun:
    """One atomic need's retrieval result, tagged with where it came from."""

    sub_question_id: str
    ain_id: int
    candidates: list[Candidate]


def sub_question_feature(sub_question_id: str) -> str:
    """Feature key marking that a sub-question's needs retrieved a document.

    `Candidate.features` is `dict[str, float]`, and the checkpoint store coerces
    every value with `float()`, so provenance has to be encoded as flat numeric
    keys rather than a set. One key per sub-question keeps it checkpoint-safe and
    survives a resumed run.
    """
    return f"ain_hit_sq{sub_question_id}"


def fuse_ain_runs(
    runs: list[AinRun],
    composite_run: list[Candidate] | None = None,
    k_const: int = 60,
    top_k: int | None = None,
    composite_weight: float = 1.0,
) -> list[Candidate]:
    """Pool per-need runs, optionally with the composite-query run.

    Features stamped on every survivor:

    `n_ain_hits` — how many distinct needs retrieved this document. A signal
    decomposition creates and the composite query structurally cannot: a
    document surfaced by six of a narrative's needs is central to it, while one
    surfaced by a single need is niche. RRF's summed score confounds "found by
    many needs" with "ranked highly by one"; only the former says the document
    spans the narrative.

    `n_sub_question_hits` and `ain_hit_sq<id>` — the same idea one level up, and
    the level that matters for answer coverage, since the narrative's distinct
    asks are its sub-questions rather than its needs.

    `from_composite` — 1.0 when the whole-narrative run found it, so the two arms
    stay separable when analysing where evidence came from.
    """
    populated = [run for run in runs if run.candidates]
    ranked_lists = [run.candidates for run in populated]
    weights = [1.0] * len(ranked_lists)

    composite_docids: set[str] = set()
    if composite_run:
        composite_docids = {c.docid for c in composite_run}
        ranked_lists.append(composite_run)
        weights.append(composite_weight)

    if not ranked_lists:
        return []

    # Counted before fusion: RRF merges duplicates away, so afterwards there is
    # no way to recover which needs contributed a given document.
    ain_hits: dict[str, int] = {}
    sub_question_hits: dict[str, set[str]] = {}
    for run in populated:
        for docid in {c.docid for c in run.candidates}:
            ain_hits[docid] = ain_hits.get(docid, 0) + 1
            sub_question_hits.setdefault(docid, set()).add(run.sub_question_id)

    fused = reciprocal_rank_fusion(
        ranked_lists, k_const=k_const, top_k=top_k, weights=weights
    )
    for candidate in fused:
        covered = sub_question_hits.get(candidate.docid, set())
        candidate.features["n_ain_hits"] = float(ain_hits.get(candidate.docid, 0))
        candidate.features["n_sub_question_hits"] = float(len(covered))
        candidate.features["from_composite"] = float(candidate.docid in composite_docids)
        for sub_question_id in covered:
            candidate.features[sub_question_feature(sub_question_id)] = 1.0
    return fused
