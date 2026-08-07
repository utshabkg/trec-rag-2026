"""Choose which pooled documents the generator is allowed to cite.

Taking the pool's top `evidence_k` in rank order is the obvious thing and the
wrong thing here. `WORK_LOG.md` §5.2 measured the project's central generation
problem: answers hold at ~373 words against a 1024-word budget and ~9.5
sentences *regardless of whether a narrative asks 3 things or 18* (correlation
-0.065 with sub-question count), and the missing material sits in documents the
answers already cite. That is a coverage failure, not a retrieval failure.

Rank-ordered selection preserves it. Fusion rewards agreement, so the documents
that rank highest are the ones many needs found — and needs cluster inside
sub-questions, so the top of the pool skews toward whichever sub-question had
the most needs. A narrative can decompose into eight sub-questions, retrieve
well for all eight, and still hand the generator ten documents about the first
two. Decomposing the query and then discarding the decomposition at the last
step would leave the original problem exactly where it was.

So evidence is selected to span sub-questions first and rank second.
"""

from __future__ import annotations

from trecrag.decomposed.fusion import sub_question_feature
from trecrag.schema import Candidate


def select_evidence(
    ranked: list[Candidate],
    sub_question_ids: list[str],
    evidence_k: int,
) -> list[Candidate]:
    """Pick `evidence_k` documents covering as many sub-questions as possible.

    Round-robin over sub-questions: each pass takes the highest-ranked unused
    document that supports the next sub-question, so every sub-question gets one
    document before any gets two. Slots left over — because a sub-question has no
    documents of its own, or because coverage is exhausted — are filled by global
    rank, which is also the whole behaviour when provenance is missing.

    Returned in pool-rank order, not selection order: the generator prompt lists
    evidence as `[0] [1] [2] …` and is told citations are ordered "strongest
    support first", so the strongest passages should still lead.
    """
    if evidence_k <= 0 or not ranked:
        return []
    if len(ranked) <= evidence_k:
        return list(ranked)

    position = {id(candidate): i for i, candidate in enumerate(ranked)}
    by_sub_question: dict[str, list[Candidate]] = {}
    for sub_question_id in sub_question_ids:
        key = sub_question_feature(sub_question_id)
        by_sub_question[sub_question_id] = [c for c in ranked if c.features.get(key)]

    selected: list[Candidate] = []
    chosen: set[str] = set()
    # Rarest coverage first: a document supporting a sub-question that only two
    # documents cover is far harder to replace than one supporting a
    # sub-question with sixty, and taking the plentiful ones first can exhaust
    # the budget before the thin sub-question is ever reached.
    order = sorted(sub_question_ids, key=lambda sq: len(by_sub_question[sq]))

    progressed = True
    while len(selected) < evidence_k and progressed:
        progressed = False
        for sub_question_id in order:
            if len(selected) >= evidence_k:
                break
            for candidate in by_sub_question[sub_question_id]:
                if candidate.docid not in chosen:
                    selected.append(candidate)
                    chosen.add(candidate.docid)
                    progressed = True
                    break

    if len(selected) < evidence_k:
        for candidate in ranked:
            if len(selected) >= evidence_k:
                break
            if candidate.docid not in chosen:
                selected.append(candidate)
                chosen.add(candidate.docid)

    return sorted(selected, key=lambda c: position[id(c)])


def effective_evidence_k(
    configured_k: int,
    n_sub_questions: int,
    per_sub_question: int = 1,
) -> int:
    """Floor `evidence_k` at what covering every sub-question actually requires.

    Deliberately a floor, never a reduction. `ARCHITECTURE.md` §4.3 held back
    raising `evidence_k` because `postprocess_rag_answer`'s word-budget
    truncation is greedy and coverage-blind, so extra evidence could mean genuine
    new facts or just more padding, and nothing could tell them apart. Widening
    on sub-question count rather than by a flat bump keeps that objection
    answered: the extra slots exist to cover asks that would otherwise have no
    document at all.

    At `per_sub_question=1` this is inert on today's data — the largest narrative
    has 8 sub-questions against a configured `evidence_k` of 10 — which is why it
    is a knob rather than a behaviour change.
    """
    return max(configured_k, n_sub_questions * max(1, per_sub_question))
