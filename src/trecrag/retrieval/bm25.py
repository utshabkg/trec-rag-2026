from __future__ import annotations

from trecrag.clients.pyserini_client import PyseriniClient
from trecrag.retrieval.base import Retriever
from trecrag.schema import Candidate, Topic


class BM25Retriever(Retriever):
    """First-stage sparse retrieval against ClimbMix via the Pyserini REST API."""

    name = "bm25"

    def __init__(self, client: PyseriniClient):
        self.client = client

    def retrieve(self, topic: Topic, k: int) -> list[Candidate]:
        # Adaptive: a handful of ClimbMix documents kill the API's response
        # stream, deterministically, for any depth that includes them. Dropping
        # to a shallower pool keeps the narrative in the submission.
        hits, actual_depth = self.client.search_adaptive(topic.text, hits=k)
        candidates = []
        for hit in hits:
            candidates.append(
                Candidate(
                    docid=hit.docid,
                    score=hit.score,
                    rank=hit.rank,
                    text=hit.text,
                    features={
                        "bm25_score": hit.score,
                        "bm25_rank": float(hit.rank),
                        # Persisted into the checkpoint, so a narrative that had to
                        # be retrieved shallowly is auditable after the fact rather
                        # than indistinguishable from a full-depth one.
                        "bm25_depth": float(actual_depth),
                    },
                )
            )
        return candidates
