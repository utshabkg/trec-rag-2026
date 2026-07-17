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
        hits = self.client.search(topic.text, hits=k)
        candidates = []
        for hit in hits:
            candidates.append(
                Candidate(
                    docid=hit.docid,
                    score=hit.score,
                    rank=hit.rank,
                    text=hit.text,
                    features={"bm25_score": hit.score, "bm25_rank": float(hit.rank)},
                )
            )
        return candidates
