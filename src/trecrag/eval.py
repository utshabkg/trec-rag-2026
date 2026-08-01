"""Retrieval metrics against graded (UMBRELA-style) qrels.

Implemented directly rather than via `pytrec_eval`, which is absent from both
conda envs here and ships sdist-only. These are the standard definitions; the
tests pin them against hand-computed values.

The metric that matters most for this pipeline is not nDCG. Fusion selects a
top-50 that the listwise reranker and the generator then consume, so what the
cascade actually needs from an earlier stage is that the right documents are
*present* -- recall@k -- with nDCG@10 mattering for the evidence the generator
sees first.

`judged_at_k` is reported alongside every run and should be read first. These
qrels were pooled from other systems' runs, so a document we retrieve may be
unjudged rather than irrelevant. Standard TREC practice scores unjudged as
non-relevant, which biases *against* any system that finds documents the pool
never saw. A comparison between two of our own configurations is still
meaningful, but an absolute nDCG here is a lower bound, not an estimate.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """Read a TREC qrels file: `topic_id iteration docid grade`."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            fields = line.split()
            if len(fields) < 4:
                continue
            topic_id, _, docid, grade = fields[0], fields[1], fields[2], fields[3]
            try:
                qrels[topic_id][docid] = int(grade)
            except ValueError:
                continue
    return dict(qrels)


def _gain(grade: int, exponential: bool) -> float:
    return (2**grade - 1) if exponential else float(grade)


def ndcg_at_k(
    ranking: list[str],
    judgements: dict[str, int],
    k: int,
    exponential: bool = True,
) -> float:
    """Normalised DCG. Unjudged documents contribute zero gain."""
    if not judgements:
        return 0.0
    dcg = sum(
        _gain(judgements.get(docid, 0), exponential) / math.log2(rank + 1)
        for rank, docid in enumerate(ranking[:k], start=1)
    )
    ideal = sorted(judgements.values(), reverse=True)[:k]
    idcg = sum(
        _gain(grade, exponential) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal, start=1)
    )
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(
    ranking: list[str], judgements: dict[str, int], k: int, relevant_grade: int = 1
) -> float:
    relevant = {d for d, g in judgements.items() if g >= relevant_grade}
    if not relevant:
        return 0.0
    return len(relevant & set(ranking[:k])) / len(relevant)


def average_precision(
    ranking: list[str], judgements: dict[str, int], relevant_grade: int = 1
) -> float:
    relevant = {d for d, g in judgements.items() if g >= relevant_grade}
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, docid in enumerate(ranking, start=1):
        if docid in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant)


def judged_at_k(ranking: list[str], judgements: dict[str, int], k: int) -> float:
    """Fraction of the top-k that appears in the qrels at all.

    Read this before trusting any other number: a low value means most of what
    we retrieved was never assessed, so the scores below are a lower bound
    rather than a measurement.
    """
    if not ranking[:k]:
        return 0.0
    return sum(1 for d in ranking[:k] if d in judgements) / len(ranking[:k])


def evaluate_run(
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    ks: tuple[int, ...] = (10, 50),
    relevant_grade: int = 1,
) -> dict[str, float]:
    """Macro-average metrics over the topics present in BOTH run and qrels.

    Scoring only the intersection keeps a partial run comparable to a complete
    one; `topics` is reported so a shrinking denominator is visible rather than
    silently flattering the average.
    """
    topic_ids = [t for t in run if t in qrels]
    if not topic_ids:
        return {"topics": 0}

    totals: dict[str, float] = defaultdict(float)
    for topic_id in topic_ids:
        ranking, judgements = run[topic_id], qrels[topic_id]
        for k in ks:
            totals[f"ndcg@{k}"] += ndcg_at_k(ranking, judgements, k)
            totals[f"recall@{k}"] += recall_at_k(ranking, judgements, k, relevant_grade)
            totals[f"judged@{k}"] += judged_at_k(ranking, judgements, k)
        totals["map"] += average_precision(ranking, judgements, relevant_grade)

    results = {name: value / len(topic_ids) for name, value in totals.items()}
    results["topics"] = len(topic_ids)
    return results
