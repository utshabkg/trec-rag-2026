"""Shared data types passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Topic:
    """One input query row from a `trec_rag_2026_queries.tsv`-style file."""

    topic_id: str
    text: str


@dataclass
class Candidate:
    """A single retrieved/reranked ClimbMix document for one topic.

    `features` accumulates per-stage signals (e.g. `bm25_score`, `bien_rank`,
    `rrf_score`) so later stages (LTR in particular) can consume upstream
    scores without re-deriving them.
    """

    docid: str
    score: float
    rank: int = 0
    text: Optional[str] = None
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class AnswerSentence:
    text: str
    citations: list[int] = field(default_factory=list)


@dataclass
class RagAnswer:
    references: list[str]
    sentences: list[AnswerSentence]
